import torch
from torch import nn


n_features = 5
batch_size = 2
a = torch.randn(batch_size, n_features)
print('a: {}'.format(a))

a = a.view(batch_size, 1, n_features)

print('a: {}'.format(a))

lin = n_features
padding = 1
kernel_size = 3
dilation = 1
stride = 1
lout = int((lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
print('Lout = {}'.format(lout))

m = nn.Conv1d(1, 20, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
out = m(a)
print('Out size {}'.format(out.size()))
print('Network: {}'.format(m))

print('Res: {}'.format(out))
