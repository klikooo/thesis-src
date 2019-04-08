import torch
import torch.nn as nn
import torch.nn.functional as F

kernel_size = 20
n_features = 1750
batch_size = 1
channels = 1

padding = int(kernel_size / 2)
pad = nn.ConstantPad1d(padding, 0)

conv = nn.Conv1d(channels, channels, kernel_size=kernel_size)
x = torch.randn(batch_size, channels, n_features)
z = conv(pad(x))
print("size: {}".format(z.size()))

f = n_features + 2 * padding - 1 * (kernel_size - 1)
# 10 + 2*2 -1 *(4-1) +1 = 10 + 4 -3 +1
print(f)
