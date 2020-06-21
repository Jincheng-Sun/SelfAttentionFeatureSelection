import sys
from Experiments.mnist.Dataloader import MNISTDataloader
from SAFS.Models import SAFSModel
import argparse
import os
import configparser

current_path = os.getcwd()


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-configs', default='configs/example.cfg', help='If use configs file to load parameters, fill the configs file path')

    opt = args.parse_args()
    config = opt.configs

    cfg = configparser.ConfigParser()
    cfg.read(config)

    # Dataloader settings
    data_path = cfg.get('Dataloader', 'data_path')
    eval_size = eval(cfg.get('Dataloader', 'eval_size'))
    shuffle = eval(cfg.get('Dataloader', 'shuffle'))
    download = eval(cfg.get('Dataloader', 'download'))
    # Model settings
    name = cfg.get('Model', 'name')
    save_path = cfg.get('Model', 'save_path')
    d_features = eval(cfg.get('Model', 'd_features'))
    d_out = eval(cfg.get('Model', 'd_out'))
    n_sub = eval(cfg.get('Model', 'n_sub'))
    n_classes = eval(cfg.get('Model', 'n_classes'))
    kernel = eval(cfg.get('Model', 'kernel'))
    stride = eval(cfg.get('Model', 'stride'))
    d_hidden = eval(cfg.get('Model', 'd_hidden'))
    d_classifier = eval(cfg.get('Model', 'd_classifier'))
    n_heads = eval(cfg.get('Model', 'n_heads'))
    random_seeds = eval(cfg.get('Model', 'random_seeds'))
    # Training settings
    lr = eval(cfg.get('Training', 'lr'))
    epochs = eval(cfg.get('Training', 'epochs'))
    batch_size = eval(cfg.get('Training', 'batch_size'))
    device = cfg.get('Training', 'device')

    dataloader = MNISTDataloader(data_path + '/', batch_size=batch_size, eval_size=eval_size,
                                 shuffle=shuffle, download=download)

    for i in range(15, 86, 10):
        d_out[-1] = i
        model = SAFSModel(name+'_out_%d'%i,
                          save_path + '/%s_out%d' % (name, i) + '/models/',
                          save_path + '/%s_out%d' % (name, i) + '/logs/',
                          d_features=d_features, d_out_list=d_out, n_subset_list=n_sub, kernel=kernel, stride=stride,
                          d_k=d_hidden, d_v=d_hidden, d_classifier=d_classifier, n_classes=n_classes, n_heads=n_heads,
                          random_seeds=random_seeds)

        model.train(epochs, lr, dataloader.train_dataloader(), dataloader.eval_dataloader(), device)


if __name__ == '__main__':
    main()
