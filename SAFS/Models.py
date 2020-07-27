import torch
from torch.optim.adamw import AdamW
from framework.model import Model
from SAFS.Modules import SelfAttentionFeatureSelection, SelfAttentionFeatureSelection_V2, LinearClassifier
from SAFS.Losses import mse_loss, cross_entropy_loss, accuracy, precision_recall, Evaluator
from SAFS.ShuffleAlgorithms import cross_shuffle, random_shuffle
from utils import TrainingControl, EarlyStopping
from tqdm import tqdm


class SAFSModel(Model):
    def __init__(
            self, name, model_path, log_path, d_features, d_out_list, n_subset_list, kernel, stride, d_k=32, d_v=32,
            n_heads=3,
            d_classifier=128, n_classes=10, f_shuffle=random_shuffle, random_seeds=[1, 2, 3], threshold=None,
            optimizer=None, no_log=False, test=False):
        if test:
            self.model_path = model_path
        else:
            super().__init__(name, model_path, log_path)
        self.n_classes = n_classes
        self.threshold = threshold

        # ----------------------------- Model ------------------------------ #

        self.model = SelfAttentionFeatureSelection_V2(f_shuffle=f_shuffle, d_features=d_features,
                                                      n_subset_list=n_subset_list, d_out_list=d_out_list,
                                                      kernel=kernel, stride=stride, d_k=d_k, d_v=d_v, h=n_heads,
                                                      random_seeds=random_seeds)

        # --------------------------- Classifier --------------------------- #

        self.classifier = LinearClassifier(d_out_list[-1], d_classifier, n_classes)

        # ------------------------------ CUDA ------------------------------ #
        self.CUDA_AVAILABLE = self.check_cuda()
        if self.CUDA_AVAILABLE:
            self.data_parallel()

        # ---------------------------- Parameters -------------------------- #
        self.parameters = list(self.model.parameters()) + list(self.classifier.parameters())
        self.optimizer = None if optimizer is None else optimizer

        # ------------------------ training control ------------------------ #
        self.controller = TrainingControl(max_step=10000, evaluate_every_nstep=100, print_every_nstep=10)
        self.early_stopping = EarlyStopping(patience=10)

        # --------------------- logging and tensorboard -------------------- #
        if not test:
            self.set_logger()
            self.set_summary_writer()
        self.no_log = no_log
        # ---------------------------- END INIT ---------------------------- #
        self.count_parameters()

    def checkpoint(self, step):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': step}
        return checkpoint

    def train_epoch(self, train_dataloader, eval_dataloader, device, smothing, earlystop):
        ''' Epoch operation in training phase'''

        if device == 'cuda':
            assert self.CUDA_AVAILABLE

        batch_counter = 0

        # update param per batch
        for batch in tqdm(
                train_dataloader, mininterval=1,
                desc='  - (Training)   ', leave=False):  # training_data should be a iterable

            # Set model and classifier training mode
            self.model.train()
            self.classifier.train()

            # get data from dataloader
            features, labels = map(lambda x: x.to(device), batch)

            batch_size = len(features)

            # forward
            self.optimizer.zero_grad()
            logits, attn = self.model(features)
            logits = logits.view(batch_size, -1)
            logits = self.classifier(logits)

            # Judge if it's a regression problem
            if self.n_classes == 1:
                pred = logits.sigmoid()
                loss = mse_loss(pred, labels)

            else:
                pred = logits
                loss = cross_entropy_loss(pred, labels, smoothing=smothing)

            # calculate gradients
            loss.backward()

            # update parameters
            self.optimizer.step()

            # get metrics for logging
            acc = accuracy(pred, labels, threshold=self.threshold)
            # precision, recall, precision_avg, recall_avg = precision_recall(pred, labels, self.n_classes,
            #                                                                 threshold=self.threshold)
            batch_counter += 1

            # training control
            state_dict = self.controller(batch_counter)

            if state_dict['step_to_print']:
                if not self.no_log:
                    self.train_logger.info(
                        '[TRAINING]   - step: %5d, loss: %3.4f, acc: %1.4f' % (
                            state_dict['step'], loss, acc))
                self.summary_writer.add_scalar('loss/train', loss, state_dict['step'])
                self.summary_writer.add_scalar('acc/train', acc, state_dict['step'])

            if state_dict['step_to_evaluate']:
                stop = self.eval_epoch(eval_dataloader, device, state_dict['step'])
                state_dict['step_to_stop'] = stop

                if earlystop & stop:
                    break

            if self.controller.current_step == self.controller.max_step:
                state_dict['step_to_stop'] = True
                break

        return state_dict

    def eval_epoch(self, dataloader, device, step=0):
        ''' Epoch operation in evaluation phase '''
        if device == 'cuda':
            assert self.CUDA_AVAILABLE

        # Set model and classifier training mode
        self.model.eval()
        self.classifier.eval()

        # use evaluator to calculate the average performance
        # evaluator = Evaluator()

        pred_list = []
        real_list = []
        total_loss = []
        with torch.no_grad():

            for batch in tqdm(
                    dataloader, mininterval=5,
                    desc='  - (Evaluation)   ', leave=False):  # training_data should be a iterable

                # get data from dataloader
                features, labels = map(lambda x: x.to(device), batch)
                batch_size = len(features)

                # get logits
                logits, attn = self.model(features)
                logits = logits.view(batch_size, -1)
                logits = self.classifier(logits)

                if self.n_classes == 1:
                    pred = logits.sigmoid()
                    loss = mse_loss(pred, labels)

                else:
                    pred = logits
                    loss = cross_entropy_loss(pred, labels, smoothing=False)

                pred_list += pred.tolist()
                real_list += labels.tolist()
                total_loss += [loss.item()]

            pred_list = torch.tensor(pred_list)
            real_list = torch.tensor(real_list)
            acc = accuracy(pred_list, real_list, threshold=self.threshold)
            loss_avg = sum(total_loss) / len(total_loss)

            if not self.no_log:
                self.eval_logger.info(
                    '[EVALUATION] - step: %5d, loss: %3.4f, acc: %1.4f' % (
                        step, loss_avg, acc))
            self.summary_writer.add_scalar('loss/eval', loss_avg, step)
            self.summary_writer.add_scalar('acc/eval', acc, step)

            state_dict = self.early_stopping(loss_avg)

            if state_dict['save']:
                checkpoint = self.checkpoint(step)
                self.save_model(checkpoint, self.model_path + self.name + '-step-%d_loss-%.5f' % (step, loss_avg))

            return state_dict['break']

    def train(self, max_epoch, lr, train_dataloader, eval_dataloader, device,
              smoothing=False, earlystop=False, save_mode='best'):
        assert save_mode in ['all', 'best']
        if self.optimizer is None:
            self.set_optimizer(AdamW, lr=lr, betas=(0.9, 0.999), weight_decay=0.001)

        # train for n epoch
        for epoch_i in range(max_epoch):
            print('[ Epoch', epoch_i, ']')
            # set current epoch
            self.controller.set_epoch(epoch_i + 1)
            # train for on epoch
            state_dict = self.train_epoch(train_dataloader, eval_dataloader, device, smoothing, earlystop)

        checkpoint = self.checkpoint(state_dict['step'])

        self.save_model(checkpoint, self.model_path + self.name + '-step-%d' % state_dict['step'])

        self.train_logger.info(
            '[INFO]: Finish Training, ends with %d epoch(s) and %d batches, in total %d training steps.' % (
                state_dict['epoch'] - 1, state_dict['batch'], state_dict['step']))

    def predict_dataset(self, data_loader, device, max_batches=None, activation=None):

        pred_list = []
        real_list = []
        attn_list = []

        self.model.eval()
        self.classifier.eval()

        batch_counter = 0

        with torch.no_grad():
            for batch in tqdm(
                    data_loader,
                    desc='  - (Testing)   ', leave=False):

                features, labels = map(lambda x: x.to(device), batch)

                # get logits
                logits, attn = self.model(features)
                logits = logits.view(logits.shape[0], -1)
                logits = self.classifier(logits)

                # Whether to apply activation function
                if activation != None:
                    pred = activation(logits)
                else:
                    pred = logits.softmax(dim=-1)
                pred_list += pred.tolist()
                real_list += labels.tolist()
                attn_list += attn

                if max_batches != None:
                    batch_counter += 1
                    if batch_counter >= max_batches:
                        break

        pred_list = torch.tensor(pred_list)
        real_list = torch.tensor(real_list)
        acc = accuracy(pred_list, real_list, threshold=self.threshold)

        return pred_list, real_list, acc, attn_list

    # def predict_batch(self, data, device, activation=None):
    #
    #     self.model.eval()
    #     self.classifier.eval()
    #
    #     with torch.no_grad():
    #         features, labels = map(lambda x: x.to(device), data)
    #
    #         # get logits
    #         logits, attn = self.model(features)
    #         logits = logits.view(logits.shape[0], -1)
    #         logits = self.classifier(logits)
    #
    #         # Whether to apply activation function
    #         if activation != None:
    #             pred = activation(logits)
    #         else:
    #             pred = logits.softmax(dim=-1)
    #
    #
    #     return pred, labels, attn
