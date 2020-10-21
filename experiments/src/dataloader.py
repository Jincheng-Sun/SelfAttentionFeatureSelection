import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        dataset = None
        self.x = None
        self.y = None
        self.N_CLS = None
        self.N_FD = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (self.x[index, :], self.y[index, :])

        return sample

    def get_feature_dim(self):
        return self.N_FD

    def get_classes(self):
        return self.N_CLS


class BaseDataloader():
    def __init__(self, train_set, test_set=None, batch_size=100, eval_size=0.1, shuffle=True):

        # Load dataset
        if test_set is None:
            n_samples = len(train_set)
            # Create indices
            indices = list(range(n_samples))
            # 20% for test set
            test_split = int(np.floor(0.2 * n_samples))
            # Split train-test indices
            train_indices, test_indices = indices[:-test_split], indices[-test_split:]
            # Split train-eval indices
            eval_split = int(np.floor(0.8 * n_samples * eval_size))
            train_indices, eval_indices = train_indices[:-eval_split], train_indices[-eval_split:]

        else:
            # Get train set size
            n_train_samples = len(train_set)
            # Create train set indices
            train_indices = list(range(n_train_samples))
            eval_split = int(np.floor(eval_size * n_train_samples))
            train_indices, eval_indices = train_indices[:-eval_split], train_indices[-eval_split:]
            # Get test set size
            n_test_samples = len(test_set)
            # Create test set indices
            test_indices = list(range(n_test_samples))

        if shuffle:
            # Shuffle train set
            np.random.seed(42)
            np.random.shuffle(train_indices)

        # Create sampler
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        eval_sampler = SubsetRandomSampler(eval_indices)

        # Create dataloaders
        self.train_loader = DataLoader(train_set, batch_size, sampler=train_sampler)
        self.eval_loader = DataLoader(train_set, batch_size, sampler=eval_sampler)
        if test_set is None:
            self.test_loader = DataLoader(train_set, batch_size, sampler=test_sampler)
        else:
            self.test_loader = DataLoader(test_set, batch_size)

    def train_dataloader(self):
        return self.train_loader

    def eval_dataloader(self):
        return self.eval_loader

    def test_dataloader(self):
        return self.test_loader
