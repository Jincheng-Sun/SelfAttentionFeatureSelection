import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class FraudDataset(Dataset):
    def __init__(self, features, labels):
        super(FraudDataset, self).__init__()

        # load data
        self.features = np.load(features)
        self.y = np.load(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (torch.from_numpy(self.features[index, :]).float(),
                  torch.from_numpy(self.y[index, :]).float())

        return sample

    def get_feature_dim(self):
        return self.features.shape[-1]


class FraudDataloader():
    def __init__(self, file_dir, batch_size, val_size, shuffle=True):

        assert val_size <= 1
        # Load training data
        train_set = FraudDataset(file_dir + 'X_train.npy', file_dir + 'y_train.npy')
        self.dims = train_set.get_feature_dim()
        n_train_samples = len(train_set)
        train_indices = list(range(n_train_samples))

        # Load test data
        test_set = FraudDataset(file_dir + 'X_test.npy', file_dir + 'y_test.npy')
        n_test_samples = len(test_set)
        test_indices = list(range(n_test_samples))
        split = int(np.floor(val_size * n_test_samples))

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(train_indices)

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

    def get_feature_dim(self):
        return self.dims
