import torch
import torch.nn as nn

class GELoss(nn.Module):
    def __init__(self):
        super(GELoss, self).__init__()

    def forward(self, x, y):
        sortedY, indecesY = torch.sort(y)
        sorted, indices = torch.sort(x)
        return 0



