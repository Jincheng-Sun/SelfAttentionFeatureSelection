import sys

sys.path.append('/home/oem/Projects/SelfAttentionFeatureSelection')
from Experiments.isolet.Dataloader import IsoletDataloader
from SAFS.Models import SAFSModel
from SAFS.utils import k_folds
import argparse
import os
import configparser

current_path = os.getcwd()


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-configs', default=None, help='If use configs file to load parameters, fill the configs file path')

    opt = args.parse_args()
    config = opt.config

    cfg = configparser.ConfigParser()
    cfg.read(config)

    # Dataloader settings
    data_path = cfg.get('Dataloader', 'data_path')
    k_fold = eval(cfg.get('Dataloader', 'k_fold'))
    n_samples = eval(cfg.get('Dataloader', 'n_samples'))
    eval_size = eval(cfg.get('Dataloader', 'eval_size'))
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

    for test_indices, fold_idx in k_folds(k_fold, n_samples):
        dataloader = IsoletDataloader(data_path, test_indices, batch_size=batch_size, eval_size=eval_size)

        model = SAFSModel(name + '_fold%d_out%d' % (fold_idx, d_out[-1]),
                          save_path + '/%s_fold%d_out%d' % (name, fold_idx, d_out[-1]) + '/models/',
                          save_path + '/%s_fold%d_out%d' % (name, fold_idx, d_out[-1]) + '/logs/',
                          d_features=d_features, d_out_list=d_out, n_subset_list=n_sub, kernel=kernel, stride=stride,
                          d_k=d_hidden, d_v=d_hidden, d_classifier=d_classifier, n_classes=n_classes, n_heads=n_heads,
                          random_seeds=random_seeds)

        model.train(epochs, lr, dataloader.train_dataloader(), dataloader.eval_dataloader(), device)


if __name__ == '__main__':
    main()
