import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util import load_ascad, test_model, HW


from util import device


class DenseSpreadNet(nn.Module):
    def __init__(self, spread_factor, input_shape, out_shape):
        super(DenseSpreadNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 100 * spread_factor).to(device)
        self.fc2 = nn.Linear(100 * spread_factor, 100 * spread_factor).to(device)
        self.spread_factor = spread_factor
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.fc3 = nn.Linear(100 * spread_factor, out_shape).to(device)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x)).to(device)
        x = F.relu(self.fc2(x)).to(device)

        x = self.fc3(x).to(device)
        # return F.softmax(x, dim=1).to(device)
        return x

    def name(self):
        return DenseSpreadNet.basename()

    @staticmethod
    def basename():
        return DenseSpreadNet.__name__

    @staticmethod
    def save_name(args):
        return "{}".format(DenseSpreadNet.basename())

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'sf': self.spread_factor,
            'out_shape': self.out_shape,
            'input_shape': self.input_shape
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = DenseSpreadNet(checkpoint['sf'], input_shape=checkpoint['input_shape'], out_shape=checkpoint['out_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return DenseSpreadNet(spread_factor=args['sf'], out_shape=args['n_classes'], input_shape=args['input_shape'])


