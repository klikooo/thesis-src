from torch.utils.data import Dataset
import torch
import numpy as np
from util import device


class DataAscad(Dataset):
    def __init__(self, x_profiling, y_profiling, train_size):
        self.train_size = train_size

        self.x_profiling = torch.from_numpy(x_profiling[:train_size].astype(np.float32)).to(device)
        self.y_profiling = torch.from_numpy(y_profiling[:train_size].astype(np.long)).to(device)

    def __len__(self):
        return self.train_size

    def __getitem__(self, index):
        return self.x_profiling[index], self.y_profiling[index]




