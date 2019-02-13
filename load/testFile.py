import torch
from torch import nn
import numpy as np

n_features = 4
batch_size = 2
a = torch.randn(batch_size, n_features)
print(a)
print(torch.cat([a, a], 1))

size = (batch_size, 1, n_features)
a = torch.arange(1.0, batch_size*n_features+1).view(size)

# a = torch.tensor([[1.0, 2.0, 3.0],
#                  [4.0, 5.0, 6.0]])
# a = a.view(batch_size, 1, n_features)
print(a)
# a= torch.tensor()

weights = torch.tensor([[[1., 1., 1.]],
                        [[0., 1., 1.]],
                        [[.0, 1., .0]]
                        ])
# print('size weights {}'.format(weights.size()[0]))
biases = torch.zeros(weights.size()[0])
print('Weights size {}, dimension: {}'.format(weights.size(), weights.dim()))
# print(weights)


conv = nn.functional.conv1d(a, weights, bias=biases, padding=1)
print('Conv out:\n{}'.format(conv))

max1 = nn.functional.max_pool1d(conv,  1)
print('Max pool out:\n{}'.format(max1))

weights2 = torch.tensor([
    [[1., 1., 1.],
     [.5, .0, .0],
     [.0, .5, .0]],

    [[0., 1., 1.],
     [.5, .0, .0],
     [.0, .5, .0]],
    ])
conv2 = nn.functional.conv1d(max1, weights2)
print('Conv2:\n{}'.format(conv2))


reshape = max1.view(batch_size, -1)
print('Reshape:\n{}'.format(reshape))
