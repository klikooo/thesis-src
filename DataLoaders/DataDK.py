from torch.utils.data import Dataset
import torch
import numpy as np
from util import device


class DataDK(Dataset):
    def __init__(self, x, y, plain, train_size):
        self.train_size = train_size

        self.x = torch.from_numpy(x[:train_size].astype(np.float32)).to(device)
        self.y = torch.from_numpy(y[:train_size].astype(np.long)).to(device)

        self.plain = torch.from_numpy(plain[:train_size]).to(device)

    def __len__(self):
        return self.train_size

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.plain[index]




