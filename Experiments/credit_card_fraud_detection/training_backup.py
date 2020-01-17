from Experiments.credit_card_fraud_detection.Dataloader import FraudDataloader
from SAFS.Models import SAFSModel
import torch.nn as nn

dataloader = FraudDataloader('data/', batch_size=256, val_size=0.1)

model = SAFSModel('test', 'models/', 'logs/', d_features=30, d_out_list=[30, 30, 10, 10], kernel=3, stride=3,
                  d_classifier=128, n_classes=2)

# model.train(200, dataloader.train_dataloader(), dataloader.val_dataloader(), 'cuda')

model.load_model('models/test1-step-1800_loss-0.00270')

pred, real = model.predict_dataset(dataloader.test_dataloader(), 'cuda')

import numpy as np
pred_ = np.array(pred)[:, 1]
real = np.array(real).astype(int)
from Experiments.evaluation_metrics import precision_recall, plot_pr_curve

area, precisions, recalls, thresholds = precision_recall(pred_, real)
plt = plot_pr_curve(recalls, precisions, auc=area)
plt.show()
plt.cla

from Experiments.evaluation_metrics import auc_roc, plot_roc_curve
auc, fprs, tprs, thresholds = auc_roc(pred_, real)
plt = plot_roc_curve(fprs, tprs, auc)
plt.show()

