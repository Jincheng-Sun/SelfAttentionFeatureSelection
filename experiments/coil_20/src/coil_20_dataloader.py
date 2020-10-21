import torch
import scipy.io
from experiments.src.dataloader import BaseDataset, BaseDataloader


class Coil20Dataset(BaseDataset):
    def __init__(self, file_dir):
        super().__init__()

        dataset = scipy.io.loadmat(file_dir)

        # load data
        self.x = torch.from_numpy(dataset['X']).float()
        self.y = torch.from_numpy(dataset['Y']).reshape(-1, 1) - 1
        self.N_CLS = 20
        self.N_FD = 1024
        print(f'classes:')
        print(set(self.y.numpy().flatten()))


class Coil20Dataloader(BaseDataloader):
    def __init__(self, file_dir, batch_size, eval_size, shuffle=True):
        dataset = Coil20Dataset(file_dir)
        super().__init__(dataset, None, batch_size, eval_size, shuffle)
