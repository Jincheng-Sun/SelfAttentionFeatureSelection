import torch
import scipy.io
from experiments.src.dataloader import BaseDataset, BaseDataloader


class IsoletDataset(BaseDataset):
    def __init__(self, file_dir):
        super().__init__()
        dataset = scipy.io.loadmat(file_dir)

        # load data
        self.x = torch.from_numpy(dataset['X']).float()
        self.y = torch.from_numpy(dataset['Y']) - 1  # shift 1
        self.N_CLS = 26
        self.N_FD = 617
        print(f'classes:')
        print(set(self.y.numpy().flatten()))


class IsoletDataloader(BaseDataloader):
    def __init__(self, file_dir, batch_size, eval_size, shuffle=True):
        dataset = IsoletDataset(file_dir)
        super().__init__(dataset, None, batch_size, eval_size, shuffle)
