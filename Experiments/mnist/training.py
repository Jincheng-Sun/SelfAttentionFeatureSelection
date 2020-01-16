from Experiments.mnist.Dataloader import MNISTDataloader
from SAFS.Models import SAFSModel
import argparse
import os

current_path = os.getcwd()


def main():
    args = argparse.ArgumentParser()
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

    assert opt.device in ['cuda', 'cpu']

    dataloader = MNISTDataloader(opt.data_path + '/', batch_size=opt.batch_size, val_size=opt.val_size,
                                 shuffle=opt.shuffle, download=opt.download)

    model = SAFSModel(opt.name, opt.save_path + '/models/', opt.save_path + '/logs/',
                      d_features=opt.d_features, d_out_list=opt.l_output, kernel=opt.kernel, stride=opt.stride,
                      d_k=opt.d_hidden, d_v=opt.d_hidden, d_classifier=opt.d_classifier, n_classes=opt.n_classes)

    model.train(opt.epochs, opt.lr, dataloader.train_dataloader(), dataloader.val_dataloader(), opt.device)


if __name__ == '__main__':
    main()
