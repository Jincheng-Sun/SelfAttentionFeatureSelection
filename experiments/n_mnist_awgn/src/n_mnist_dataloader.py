import torch
import scipy.io
import numpy as np
from experiments.src.dataloader import BaseDataset, BaseDataloader


class NMnistDataset(BaseDataset):
    def __init__(self, file_dir, train=True):
        super().__init__()
        dataset = scipy.io.loadmat(file_dir)

        if train:
            # Load train set
            self.x = torch.from_numpy(dataset['train_x']).view(-1, 28 * 28).type(torch.float32) / 255
            self.y = torch.from_numpy(np.argmax(dataset['train_y'], axis=1).reshape(-1, 1))
        else:
            # Load test set
            self.x = torch.from_numpy(dataset['test_x']).view(-1, 28 * 28).type(torch.float32) / 255
            self.y = torch.from_numpy(np.argmax(dataset['test_y'], axis=1).reshape(-1, 1))

        self.N_CLS = 10
        self.N_FD = 784
        print(f'classes:')
        print(set(self.y.numpy().flatten()))


class NMnistDataloader(BaseDataloader):
    def __init__(self, file_dir, batch_size=100, eval_size=0.1, shuffle=True):
        # Load datasets
        train_set = NMnistDataset(file_dir, train=True)
        test_set = NMnistDataset(file_dir, train=False)
        super().__init__(train_set, test_set, batch_size, eval_size, shuffle)
