import torch
import torch.nn as nn




n_features = 10
batch_size = 1
channels = 1

mlp1 = nn.Linear(n_features, 10)

x = torch.ones(batch_size, n_features)

print(mlp1(x))


print(mlp1.weight)