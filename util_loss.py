import argparse

import torch
import torch.nn as nn

from models.ConvNetKernel import ConvNetKernel

loss_function_map = {
    "CrossEntropy": nn.CrossEntropyLoss(),
    "L1": nn.L1Loss(),
    "MSE": nn.MSELoss(),
    "NLL": nn.NLLLoss(),
}


optimizer_function_map = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
}


parser = argparse.ArgumentParser('Train a nn on the ascad db')
parser.add_argument("-a", "--ab", nargs="+")
args = parser.parse_args()
li = args.ab
d = {}
it = iter(list(li))
for x, y in zip(*[iter(li)]*2):
    d[x] = float(y)


model = ConvNetKernel(3500, 256, 5)
d["params"] = model.parameters()
print(d)
opti = optimizer_function_map['Adam'](**d)
print(opti)
