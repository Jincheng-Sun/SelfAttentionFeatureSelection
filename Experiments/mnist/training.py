from Experiments.mnist.Dataloader import MNISTDataloader
from SAFS.Models import SAFSModel
import torch.nn as nn

dataloader = MNISTDataloader('data/', batch_size=256, val_size=0.1)

model = SAFSModel('models/test1', 'logs/test1', d_features=784, d_out_list=[196, 196, 49, 49], kernel=4, stride=3,
                  d_classifier=128, d_output=10)

model.train(200, dataloader.train_dataloader(), dataloader.val_dataloader(), 'cuda')

# model.load_model('models/test1-step-1800_loss-0.00270')

# pred, real = model.get_predictions(dataloader.test_dataloader(), 'cuda')
#
# import numpy as np
# pred_ = np.array(pred)[:, 1]
# real = np.array(real).astype(int)
# from utils import precision_recall, plot_pr_curve
#
# area, precisions, recalls, thresholds = precision_recall(pred_, real)
# plt = plot_pr_curve(recalls, precisions, auc=area)
# plt.show()
# plt.cla
#
# from utils import auc_roc, plot_roc_curve
# auc, fprs, tprs, thresholds = auc_roc(pred_, real)
# plt = plot_roc_curve(fprs, tprs, auc)
# plt.show()

