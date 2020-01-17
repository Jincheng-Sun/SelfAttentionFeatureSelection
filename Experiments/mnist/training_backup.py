from Experiments.mnist.Dataloader import MNISTDataloader
from SAFS.Models import SAFSModel
from SAFS.ShuffleAlgorithms import cross_shuffle

import torch.nn as nn

dataloader = MNISTDataloader('data/', batch_size=64, val_size=0.1)

# model = SAFSModel('test1', 'models/', 'logs/', d_features=784, d_out_list=[196, 196, 49, 49], kernel=4, stride=3,
#                   d_classifier=128, d_output=10)

# model = SAFSModel('test3-stride4', 'models/', 'logs/', d_features=784, d_out_list=[196, 196, 49, 49], kernel=4, stride=4,
#                   d_classifier=128, d_output=10)

# model = SAFSModel('test3-stride4', 'models/', 'logs/', d_features=784, d_out_list=[784, 784, 784, 784], kernel=4, stride=4,
#                   d_classifier=128, n_classes=10, n_replica=1)

# model = SAFSModel('test4-rep8', 'models/', 'logs/', d_features=784, d_out_list=[784, 784, 784, 784], kernel=4, stride=4,
#                   d_classifier=128, n_classes=10, n_replica=8)

# model = SAFSModel('test5-1layer', 'models/', 'logs/', d_features=784, d_out_list=[784], kernel=4, stride=4,
#                   d_classifier=128, n_classes=10, n_replica=8)

model = SAFSModel('test6-1layer', 'models/', 'logs/', d_features=784, d_out_list=[49], kernel=4, stride=4,
                  d_classifier=128, n_classes=10, n_replica=8)

model.train(200, dataloader.train_dataloader(), dataloader.val_dataloader(), 'cuda')

# model.load_model('test4-rep8-step-19100_loss-0.05436')


# pred, real = model.predict_dataset(dataloader.test_dataloader(), 'cuda')
# import numpy as np
# pred = np.argmax(np.array(pred), axis=1)
# real = np.array(real).astype(int)
#
# from Experiments.evaluation_metrics import multiclass_metrics, plot_cm
# acc, cm, report = multiclass_metrics(real, pred)
# print(report)
# plt = plot_cm(cm)
#
# plt.show()

#
# Visualization of the attention

from Experiments.visualization import visualize_MNIST_from_batch, visualize_MNIST_by_class

import seaborn as sn
import matplotlib.pyplot as plt

pred, real, attn = model.predict_batch(dataloader.test_dataloader().dataset[:50], 'cuda')

# map, label = visualize_MNIST_from_batch(1, attn, real, cross_shuffle, 8, 4, 3)
# sn.heatmap(map.reshape(28, 28))
# print(label)
# plt.show()
# plt.clf()

def plot(cls=3):
    maps, _ = visualize_MNIST_by_class(cls, attn, real, cross_shuffle, 8, 4, 4)

    for map in maps:
        sn.heatmap(map)
        print(map)
        plt.show()
        plt.clf()
