import torch
from torchvision.datasets import FashionMNIST
from experiments.src.dataloader import BaseDataset, BaseDataloader


class FashionMNISTDataset(BaseDataset):
    def __init__(self, file_dir, train, download):
        super().__init__()
        dataset = FashionMNIST(file_dir, train=train, download=download)
        # load data
        self.x = dataset.data.view(-1, 28 * 28).type(torch.float32) / 255
        self.y = dataset.targets.view(-1, 1)
        self.N_CLS = dataset.classes
        self.N_FD = 784
        print(f'classes:')
        print(set(dataset.targets.numpy().flatten()))


class FashionMNISTDataloader(BaseDataloader):
    def __init__(self, file_dir, batch_size, eval_size, shuffle=True, download=True):
        train_set = FashionMNISTDataset(file_dir, train=True, download=download)
        test_set = FashionMNISTDataset(file_dir, train=False, download=download)
        super().__init__(train_set, test_set, batch_size, eval_size, shuffle)
