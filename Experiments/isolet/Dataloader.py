import torch
import scipy.io
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class IsoletDataset(Dataset):
    def __init__(self, file_dir):
        super(IsoletDataset, self).__init__()

        dataset = scipy.io.loadmat(file_dir)

        # load data
        self.input = torch.from_numpy(dataset['X']).float()
        self.labels = torch.from_numpy(dataset['Y']).float() - 1  # shift 1

    def __len__(self):
        return self.labels.size()[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (self.input[index, :], self.labels[index, :])

        return sample

    def get_feature_dim(self):
        return self.input.shape[-1]


# class IsoletDataloader():
#     def __init__(self, file_dir, batch_size, train_size, test_size, shuffle=True):
#         assert train_size <= 1
#         # Load training data
#         dataset = IsoletDataset(file_dir)
#         n_samples = len(dataset)
#         indices = list(range(n_samples))
#
#         train_split = int(np.floor(train_size * n_samples))
#         test_split = int(np.floor(test_size * n_samples))
#
#         if shuffle:
#             np.random.seed(42)
#             np.random.shuffle(indices)
#
#         train_indices, eval_indices, test_indices = indices[:train_split], indices[train_split:-test_split], indices[
#                                                                                                              -test_split:]
#         train_sampler = SubsetRandomSampler(train_indices)
#         eval_sampler = SubsetRandomSampler(eval_indices)
#         test_sampler = SubsetRandomSampler(test_indices)
#
#         self.train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)
#         self.eval_loader = DataLoader(dataset, batch_size, sampler=eval_sampler)
#         self.test_loader = DataLoader(dataset, batch_size, sampler=test_sampler)
#
#     def train_dataloader(self):
#         return self.train_loader
#
#     def eval_dataloader(self):
#         return self.eval_loader
#
#     def test_dataloader(self):
#         return self.test_loader

class IsoletDataloader():
    def __init__(self, file_dir, batch_size, test_size=0.2, eval_size=0.1):
        # Load training data
        dataset = IsoletDataset(file_dir)
        n_samples = len(dataset)
        indices = list(range(n_samples))

        # train_indices = [i for i in indices if not i in test_indices]
        test_split = int(np.floor(test_size * len(indices)))
        train_indices, test_indices = indices[:-test_split], indices[-test_split:]
        eval_split = int(np.floor(eval_size * len(indices)))
        train_indices, eval_indices = train_indices[:-eval_split], train_indices[-eval_split:]

        train_sampler = SubsetRandomSampler(train_indices)
        eval_sampler = SubsetRandomSampler(eval_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        self.train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)
        self.eval_loader = DataLoader(dataset, batch_size, sampler=eval_sampler)
        self.test_loader = DataLoader(dataset, batch_size, sampler=test_sampler)

    def train_dataloader(self):
        return self.train_loader

    def eval_dataloader(self):
        return self.eval_loader

    def test_dataloader(self):
        return self.test_loader
