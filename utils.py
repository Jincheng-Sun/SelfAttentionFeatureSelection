import logging
import numpy as np
class logger():
    def __init__(self, logger_name, log_path, mode, level=logging.INFO, format = "%(asctime)s - %(message)s"):
        self.logging = logging.getLogger(logger_name)
        self.logging.setLevel(level)
        fh = logging.FileHandler(log_path, mode)
        fh.setLevel(level)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter(format)
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        self.logging.addHandler(fh)
        self.logging.addHandler(sh)

    def info(self, msg):
        return self.logging.info(msg)

class TrainingControl():
    def __init__(self, max_step, evaluate_every_nstep, print_every_nstep):
        self.state_dict = {
            'epoch': 0,
            'batch': 0,
            'step': 0,
            'step_to_evaluate': False,
            'step_to_print': False,
            'step_to_stop': False
        }
        self.max_step = max_step
        self.eval_every_n = evaluate_every_nstep
        self.print_every_n = print_every_nstep
        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0

    def __call__(self, batch):
        self.current_step += 1
        self.state_dict['batch'] = batch
        self.state_dict['step'] = self.current_step
        self.state_dict['step_to_evaluate'] = np.equal(np.mod(self.current_step, self.eval_every_n), 0)
        self.state_dict['step_to_print'] = np.equal(np.mod(self.current_step, self.print_every_n), 0)
        self.state_dict['step_to_stop'] = np.equal(self.current_step, self.max_step)
        return self.state_dict

    def set_epoch(self, epoch):
        self.state_dict['epoch'] = epoch

    def reset_state(self):
        self.state_dict = {
            'epoch': 0,
            'batch': 0,
            'step': 0,
            'step_to_evaluate': False,
            'step_to_print': False,
            'step_to_stop': False
        }
        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0


class EarlyStopping():
    def __init__(self, patience, mode='best'):
        self.patience = patience
        self.mode = mode
        self.best_loss = 9999
        self.waitting = 0
        self.state_dict = {
            'save': False,
            'break': False
        }

    def __call__(self, val_loss):
        self.state_dict['save'] = False
        self.state_dict['break'] = False

        if val_loss <= self.best_loss:
            self.best_loss = val_loss
            self.waitting = 0
            self.state_dict['save'] = True

        else:
            self.waitting += 1

            if self.mode == 'best':
                self.state_dict['save'] = False
            else:
                self.state_dict['save'] = True

            if self.waitting == self.patience:
                self.state_dict['break'] = True

        return self.state_dict
    def reset_state(self):
        self.best_loss = 9999
        self.waitting = 0
        self.state_dict = {
            'save': False,
            'break': False
        }

def k_folds(k, n_samples):
    indices = list(range(n_samples))
    per_fold = int(np.floor(np.divide(n_samples, k)))
    for i in range(k):
        start = int(i * per_fold)
        end = int((i + 1) * per_fold)
        if i == k - 1:
            yield indices[start:], i + 1
        else:
            yield indices[start:end], i + 1