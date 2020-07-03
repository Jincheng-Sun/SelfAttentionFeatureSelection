import torch
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class FashionMNISTDataset(Dataset):
    def __init__(self, file_dir, train, download):
        super(FashionMNISTDataset, self).__init__()

        dataset = FashionMNIST(file_dir, train=train, download=download)

        # load data
        self.images = dataset.data.view(-1, 28*28).type(torch.float32) / 255
        self.labels = dataset.targets.view(-1, 1)
        self.classes = dataset.classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (self.images[index, :], self.labels[index, :])

        return sample

    def get_feature_dim(self):
        return self.images.shape[-1]

    def get_classes(self):
        return len(self.classes)


class FashionMNISTDataloader():
    def __init__(self, file_dir, batch_size, eval_size, shuffle=True, download=True):

        assert eval_size <= 1
        # Load training data
        train_set = FashionMNISTDataset(file_dir, train=True, download=download)
        n_train_samples = len(train_set)
        train_indices = list(range(n_train_samples))
        split = int(np.floor(eval_size * n_train_samples))


        # Load test data
        test_set = FashionMNISTDataset(file_dir, train=False, download=download)
        n_test_samples = len(test_set)
        test_indices = list(range(n_test_samples))

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(train_indices)

        train_indices, eval_indices = train_indices[:-split], train_indices[-split:]

        train_sampler = SubsetRandomSampler(test_indices)
        eval_sampler = SubsetRandomSampler(eval_indices)

        self.train_loader = DataLoader(train_set, batch_size, sampler=train_sampler)
        self.eval_loader = DataLoader(train_set, batch_size, sampler=eval_sampler)
        self.test_loader = DataLoader(test_set, batch_size)


    def train_dataloader(self):
        return self.train_loader

    def eval_dataloader(self):
        return self.eval_loader

    def test_dataloader(self):
        return self.test_loader

