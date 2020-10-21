from Experiments.isolet.Dataloader import IsoletDataloader
from SAFS.Models import SAFSModel
import argparse
import os
import configparser

def main():
    args = argparse.ArgumentParser()
    args.add_argument('-configs', default=None, help='If use configs file to load parameters, fill the configs file path')
    opt = args.parse_args()
    config = opt.config

    cfg = configparser.ConfigParser()
    cfg.read(config)

    data_path = cfg.get('Dataloader', 'data_path')
    shuffle = eval(cfg.get('Dataloader', 'shuffle'))
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
    lr = eval(cfg.get('Training', 'lr'))
    epochs = eval(cfg.get('Training', 'epochs'))
    batch_size = eval(cfg.get('Training', 'batch_size'))
    val_size = eval(cfg.get('Training', 'val_size'))
    device = cfg.get('Training', 'device')


    dataloader = IsoletDataloader(data_path, batch_size=batch_size, train_size=val_size,
                                  shuffle=shuffle)

    model = SAFSModel(name, save_path + '/models/', save_path + '/logs/',
                      d_features=d_features, d_out_list=d_out, n_subset_list=n_sub, kernel=kernel, stride=stride,
                      d_k=d_hidden, d_v=d_hidden, d_classifier=d_classifier, n_classes=n_classes)

    model.train(epochs, lr, dataloader.train_dataloader(), dataloader.eval_dataloader(), device)


    current_path = os.getcwd()
    model.load_model('models/test1-step-1800_loss-0.00270')

    pred, real = model.get_predictions(dataloader.test_dataloader(), 'cuda')

    import numpy as np
    pred_ = np.array(pred)[:, 1]
    real = np.array(real).astype(int)
    from utils import precision_recall, plot_pr_curve

    area, precisions, recalls, thresholds = precision_recall(pred_, real)
    plt = plot_pr_curve(recalls, precisions, auc=area)
    plt.show()
    plt.cla

    from utils import auc_roc, plot_roc_curve
    auc, fprs, tprs, thresholds = auc_roc(pred_, real)
    plt = plot_roc_curve(fprs, tprs, auc)
    plt.show()
if __name__ == '__main__':
    main()
