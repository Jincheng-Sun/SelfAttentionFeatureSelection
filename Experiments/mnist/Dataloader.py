import torch
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class MNISTDataset(Dataset):
    def __init__(self, file_dir, train, download):
        super(MNISTDataset, self).__init__()

        dataset = MNIST(file_dir, train=train, download=download)

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


class MNISTDataloader():
    def __init__(self, file_dir, batch_size, val_size, shuffle=True, download=True):

        assert val_size <= 1
        # Load training data
        train_set = MNISTDataset(file_dir, train=True, download=download)
        n_train_samples = len(train_set)
        train_indices = list(range(n_train_samples))

        # Load test data
        test_set = MNISTDataset(file_dir, train=False, download=download)
        n_test_samples = len(test_set)
        test_indices = list(range(n_test_samples))
        split = int(np.floor(val_size * n_test_samples))

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

        test_indices, val_indices = test_indices[split:], test_indices[:split]

        test_sampler = SubsetRandomSampler(test_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(train_set, batch_size, num_workers=4)
        self.val_loader = DataLoader(test_set, batch_size, sampler=valid_sampler, num_workers=4)
        self.test_loader = DataLoader(test_set, batch_size, sampler=test_sampler, num_workers=4)


    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

