import torch
import torch.nn as nn

kernel_size = 6
n_features = 10
batch_size = 1
channels = 1
padding = int(kernel_size/2.0)

x = torch.ones(batch_size, channels, n_features)


conv1 = nn.Conv1d(channels, 3, kernel_size=kernel_size, padding=padding)
conv2 = nn.Conv1d(3, 3, kernel_size=kernel_size, padding=0)

conv1.weight.data.fill_(1.0)
conv1.bias.data.fill_(0.0)
conv2.bias.data.fill_(0.0)

weights = torch.tensor([[[1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0, 1.0]],

        [[2.0, 2.0, 2.0, 2.0, 2.0],
         [2.0, 2.0, 2.0, 2.0, 2.0],
         [2.0, 2.0, 2.0, 2.0, 2.0]],

        [[3.0, 3.0, 3.0, 3.0, 3.0],
         [3.0, 3.0, 3.0, 3.0, 3.0],
         [3.0, 3.0, 3.0, 3.0, 3.0, ]]])
conv2.weight.data = weights
weights = conv2.weight


print("Filter conv layer 1:")
print(weights)
print()
y = conv1(x)
ones = torch.ones([1, 3, 11])
print("Y output: {}".format(y))
z = conv2(y)

print("size: {}".format(z.size()))
print(z)

