import torch
import scipy.io
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class NMnistDataset(Dataset):
    def __init__(self, file_dir, train=True):
        super(NMnistDataset, self).__init__()

        dataset = scipy.io.loadmat(file_dir)

        # load data
        if train:
            self.images = torch.from_numpy(dataset['train_x']).view(-1, 28 * 28).type(torch.float32) / 255
            self.labels = torch.from_numpy(np.argmax(dataset['train_y'], axis=1).reshape(-1, 1))
            self.classes = 10
        else:
            self.images = torch.from_numpy(dataset['test_x']).view(-1, 28 * 28).type(torch.float32) / 255
            self.labels = torch.from_numpy(np.argmax(dataset['test_y'], axis=1).reshape(-1, 1))
            self.classes = 10

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
        return self.classes


class NMnistDataloader():
    def __init__(self, file_dir, batch_size, eval_size, shuffle=True):
        assert eval_size <= 1
        # Load training data
        train_set = NMnistDataset(file_dir, train=True)
        n_train_samples = len(train_set)
        train_indices = list(range(n_train_samples))
        split = int(np.floor(eval_size * n_train_samples))

        # Load test data
        test_set = NMnistDataset(file_dir, train=False)
        n_test_samples = len(test_set)
        test_indices = list(range(n_test_samples))

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(train_indices)

        train_indices, eval_indices = train_indices[:-split], train_indices[-split:]

        train_sampler = SubsetRandomSampler(train_indices)
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
