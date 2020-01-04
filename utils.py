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


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams['savefig.dpi'] = 300  # pixel
plt.rcParams['figure.dpi'] = 300  # resolution
plt.rcParams["figure.figsize"] = [5, 4] # figure size

def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = precision_recall_curve(y_true=y_test, probas_pred=y_pred)
    area = auc(recalls, precisions)
    return area, precisions, recalls, thresholds

def plot_pr_curve(recalls, precisions, auc, x_axis = 1):
    plt.rcParams['savefig.dpi'] = 300  # pixel
    plt.rcParams['figure.dpi'] = 300  # resolution
    plt.rcParams["figure.figsize"] = [5, 4]  # figure size

    plt.plot(recalls, precisions, color="darkorange", label='Precision-Recall curve (area = %0.3f)' % auc)
    plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, x_axis])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    return plt

def auc_roc(y_pred, y_test):
    auc = roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds

def plot_roc_curve(fprs, tprs, auc, x_axis = 1):

    plt.plot(fprs, tprs, color="darkorange", label='ROC curve (area = %0.3f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, x_axis])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return plt