import sys
sys.path.append('/home/oem/Projects/SelfAttentionFeatureSelection')
from Experiments.mnist.Dataloader import MNISTDataloader
from SAFS.Models import SAFSModel
import argparse
import os
import configparser

current_path = os.getcwd()


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-config', default=None, help='If use config file to load parameters, fill the config file path')
    args.add_argument('-data_path', default=current_path, help='Data file path')
    args.add_argument('-download', type=bool, default=False, help='Download MNIST dataset')
    args.add_argument('-shuffle', type=bool, default=False, help='Whether to shuffle the dataset')
    args.add_argument('-name', help='Model save name')
    args.add_argument('-save_path', default=current_path, help='Model and log save file path')
    args.add_argument('-d_features', type=int, help='Input feature(1d) dimension')
    args.add_argument('-l_outputs', nargs='+', help='Output dimension list')
    args.add_argument('-n_classes', type=int, help='Number of output classes')
    args.add_argument('-kernel', type=int, help='Kernel size')
    args.add_argument('-stride', type=int, help='Conv stride')
    args.add_argument('-d_hidden', type=int, help='Query/Key/Value hidden units')
    args.add_argument('-d_classifier', type=int, help='Classifier hidden units')
    args.add_argument('-lr', type=float, default=0.002, help='Learning rate')
    args.add_argument('-epochs', type=int, help='Training epochs')
    args.add_argument('-batch_size', type=int, help='Training batch size')
    args.add_argument('-val_size', type=float, help='Validation size')
    args.add_argument('-device', default='cuda', help='Run model on GPU if it is `cuda`, else `cpu`')
    opt = args.parse_args()

    config = opt.config
    data_path = opt.data_path
    download = opt.download
    shuffle = opt.shuffle
    name = opt.name
    save_path = opt.save_path
    d_features = opt.d_features
    l_outputs = opt.l_outputs
    n_classes = opt.n_classes
    kernel = opt.kernel
    stride = opt.stride
    d_hidden = opt.d_hidden
    d_classifier = opt.d_classifier
    lr = opt.lr
    epochs = opt.epochs
    batch_size = opt.batch_size
    val_size = opt.val_size
    device = opt.device

    # if use config file, replace the values
    if config is not None:
        cfg = configparser.ConfigParser()
        try:
            cfg.read(config)
        except:
            raise

        data_path = cfg.get('Dataloader', 'data_path')
        download = eval(cfg.get('Dataloader', 'download'))
        shuffle = eval(cfg.get('Dataloader', 'shuffle'))
        name = cfg.get('Model', 'name')
        save_path = cfg.get('Model', 'save_path')
        d_features = eval(cfg.get('Model', 'd_features'))
        l_outputs = eval(cfg.get('Model', 'l_outputs'))
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


    dataloader = MNISTDataloader(data_path + '/', batch_size=batch_size, val_size=val_size,
                                 shuffle=shuffle, download=download)

    model = SAFSModel(name, save_path + '/models/', save_path + '/logs/',
                      d_features=d_features, d_out_list=l_outputs, kernel=kernel, stride=stride,
                      d_k=d_hidden, d_v=d_hidden, d_classifier=d_classifier, n_classes=n_classes)

    model.train(epochs, lr, dataloader.train_dataloader(), dataloader.val_dataloader(), device)


if __name__ == '__main__':
    main()
