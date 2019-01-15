import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ascad import load_ascad, HW, SBOX_INV, SBOX, test_model

device = torch.device('cuda:0')

v = np.array([0.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([0.0, 4.0, 7.0, 8.0, 9.0])
z = np.array([0.0, 2.0, -1.0, 0.0, 13.0])
x = np.array([v, y, z])
xx = np.array([z, v, y])

v = np.array([1.0, 6.0, 3.0, 10.0, 9.0])
y = np.array([3.0, 4.0, 7.0, 8.0, 9.0])
print(v)
print(y)
tensor_v = Variable(torch.from_numpy(v))
tensor_y = Variable(torch.from_numpy(y))
MAX = torch.max(tensor_v, tensor_y).to(device)
print('MAX: {}'.format(MAX))

spread_factor = 2

tensor_x = Variable(torch.from_numpy(x))

input_size = tensor_x.size()
batch_size = input_size[0]
num_neurons = input_size[1]

print('Data: \n{}'.format(x))
print('Input size: {}'.format(input_size))

tensor_max, _ = tensor_x.max(dim=0)
tensor_min, _ = tensor_x.min(dim=0)
print('Min {}'.format(tensor_min))
print('Max {}'.format(tensor_max))

tensor_numerator = tensor_x - tensor_min
tensor_denominator = tensor_max - tensor_min
tensor_denominator = torch.where(torch.zeros([1]).to(device).double() == tensor_denominator.to(device),
                                 torch.ones([1]).to(device).double(),
                                 tensor_denominator.to(device)).to(device)
print('Numerator \n{}'.format(tensor_numerator))
print('Denominator\n{}'.format(tensor_denominator))

x_prime = tensor_numerator / tensor_denominator
print('In [0,1]:\n{}'.format(x_prime))
x_prime = x_prime * spread_factor
print('In [0,sf]\n{}'.format(x_prime))

x_spread = x_prime.repeat(1, spread_factor).view(batch_size * spread_factor, num_neurons)
print('Spread:{}\n{}'.format(x_spread.size(), x_spread))

x_spread = x_spread.transpose(0, 1).to(device)
print('Spread.T:{}\n{}'.format(x_spread.size(), x_spread))

centroids = torch.arange(0.5, spread_factor).to(device).unsqueeze(0).to(device)
# print('Centroids:{}'.format(centroids.unsqueeze(0)))
# centroids = Variable(torch.from_numpy(np.array([[0.5, 1.5]])))
print('Centroids:{}'.format(centroids))
# TODO: fix something in contguous
centroids = centroids.expand(batch_size * num_neurons, -1).contiguous().view(num_neurons, batch_size * spread_factor)
centroids = centroids.double()
print('Centroids size, expanded:{}\n{}'.format(centroids.size(), centroids))

absolute = 1 - (centroids - x_spread).abs()
# print('abs:\n{}'.format(absolute))

nc = torch.max(torch.zeros([1]).to(device).double(), absolute).to(device)
print('nc:\n{}'.format(nc))

result = torch.where(
    (x_spread < centroids).__and__(centroids == 0.5).__or__(
        (x_spread > centroids).__and__(centroids == spread_factor - 0.5)
    ),
    torch.ones([1]).to(device).double(),
    nc.double()
).to(device)
print('Result:\n{}'.format(result))
res = result.transpose(0, 1).contiguous().view(batch_size, num_neurons * spread_factor)
print('Res2:\n{}'.format(res))
