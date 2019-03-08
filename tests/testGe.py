import torch
import numpy as np
import math

x = [[4, 2, 6, 1],
     [1, 5, 7, 2],
     [7, 3, 2, 2]]
y = [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0]]

losses = np.arange(0, 4)
weights = [1/(1 + math.exp(-1 * (x-50)/50)) for x in losses]
weights = np.array(weights)
weights = torch.from_numpy(weights.astype(np.float32))
# print(weights)

x = np.array(x)
y = np.array(y)

x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(y.astype(np.int32))
s = torch.nn.Softmax(dim=1)
x = s(x)


sortedY, indicesY = torch.sort(y, descending=True)
sortedX, indicesX = torch.sort(x, descending=True)
indicesSbox = indicesY.t()[0]


# TODO: select the correct index for the weights and perform matrix multiplication correctly

print(sortedX)
print(torch.mv(sortedX, weights))
