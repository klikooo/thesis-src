import torch
import torch.nn as nn
import torch.nn.functional as F

from util import device


class DenseBatch(nn.Module):
    def __init__(self, spread_factor, input_shape, out_shape):
        super(DenseBatch, self).__init__()

        self.spread_factor = spread_factor
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.spread_neurons = 100 * self.spread_factor

        self.fc1 = nn.Linear(self.input_shape, 100).to(device)
        self.bn = nn.BatchNorm1d(100).to(device)
        self.fc2 = nn.Linear(100, self.spread_neurons).to(device)
        self.fc3 = nn.Linear(self.spread_neurons, self.out_shape).to(device)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x)).to(device)

        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).to(device)
        return x

    def name(self):
        return DenseBatch.basename()

    @staticmethod
    def basename():
        return DenseBatch.__name__

    @staticmethod
    def save_name(args):
        return "{}".format(DenseBatch.basename())

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

        model = DenseBatch(checkpoint['sf'], input_shape=checkpoint['input_shape'], out_shape=checkpoint['out_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return DenseBatch(spread_factor=args['sf'], out_shape=args['n_classes'], input_shape=args['input_shape'])


