import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

from util import device


class SpreadNetIn(nn.Module):
    def __init__(self, spread_factor, input_shape, out_shape):
        super(SpreadNetIn, self).__init__()
        n_hidden = 100
        np_min = np.array([np.finfo(np.float32).min] * n_hidden)
        np_max = np.array([np.finfo(np.float32).max] * n_hidden)
        self.spread_factor = spread_factor
        self.input_shape = input_shape
        self.out_shape = out_shape
        # self.tensor_min = Variable(torch.from_numpy(np_max), requires_grad=False).to(device)
        # self.tensor_max = Variable(torch.from_numpy(np_min), requires_grad=False).to(device)
        self.tensor_min = np_max
        self.tensor_max = np_min

        self.intermediate_values = []
        self.intermediate_values2 = []

        self.fc1 = nn.Linear(input_shape, n_hidden).to(device)
        self.fc3 = nn.Linear(n_hidden * spread_factor, out_shape).to(device)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.training = True

    def forward(self, x):
        # x = F.relu(self.fc1(x)).to(device)
        x = self.fc1(x).to(device)
        self.intermediate_values2.append(x.detach().cpu().numpy())

        x = self.spread(x)
        self.intermediate_values.append(x.detach().cpu().numpy())
        x = self.fc3(x).to(device)
        return x

    def spread(self, x):
        input_size = x.size()
        batch_size = input_size[0]
        num_neurons = input_size[1]

        tensor_max = Variable(torch.from_numpy(self.tensor_max), requires_grad=False).to(device)
        tensor_min = Variable(torch.from_numpy(self.tensor_min), requires_grad=False).to(device)
        if self.training:
            x_max, _ = x.max(dim=0)
            x_min, _ = x.min(dim=0)
            tensor_max = torch.max(tensor_max, x_max).to(device)
            tensor_min = torch.min(tensor_min, x_min).to(device)
            self.tensor_max = tensor_max.detach().cpu().numpy()
            self.tensor_min = tensor_min.detach().cpu().numpy()
            # print('Size x: {}'.format(input_size))

        tensor_numerator = x - tensor_min
        tensor_denominator = tensor_max - tensor_min
        # Replace 0 with a 1 so we don't divide by zero
        tensor_denominator = torch.where(torch.zeros([1]).to(device) == tensor_denominator.to(device),
                                         torch.ones([1]).to(device),
                                         tensor_denominator.to(device)).to(device)

        # Calculate x'
        x_prime = tensor_numerator / tensor_denominator
        x_prime = x_prime * self.spread_factor

        # Spread x'
        x_spread = x_prime.repeat(1, self.spread_factor).view(batch_size * self.spread_factor, num_neurons)
        x_spread = x_spread.transpose(0, 1).to(device)
        # print('diff: {}'.format(tensor_max - tensor_min))

        # Create the centroids
        centroids = torch.arange(0.5, self.spread_factor).to(device).unsqueeze(0).to(device)
        # TODO: fix something in contguous, which could make it faster
        centroids = centroids.expand(batch_size * num_neurons, -1).contiguous().view(num_neurons,
                                                                                     batch_size * self.spread_factor)
        centroids = centroids.float()

        # Calulate the part of function n_c => max(1, 1 - |c - x'|)
        absolute = 1 - (centroids - x_spread).abs()
        nc = torch.max(torch.zeros([1]).to(device).float(), absolute).to(device)

        # Calculate the whole of the nc function and reshape to the correct size
        result = torch.where(
            (x_spread < centroids).__and__(centroids == 0.5).__or__(
                (x_spread > centroids).__and__(centroids == self.spread_factor - 0.5)
            ),
            torch.ones([1]).to(device).float(),
            nc.float()
        ).to(device)
        res = result.transpose(0, 1).contiguous().view(batch_size, num_neurons * self.spread_factor)
        return res

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'min': self.tensor_min,
            'max': self.tensor_max,
            'sf': self.spread_factor,
            'out_shape': self.out_shape,
            'input_shape': self.input_shape
        }, path)

    @staticmethod
    def load_spread(file):
        checkpoint = torch.load(file)

        model = SpreadNetIn(checkpoint['sf'], input_shape=checkpoint['input_shape'], out_shape=checkpoint['out_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.tensor_max = checkpoint['max']
        model.tensor_min = checkpoint['min']
        model.training = False
        return model

    def name(self):
        return "SpreadNetIn"

    @staticmethod
    def basename():
        return "SpreadNetIn"

    @staticmethod
    def save_name(args):
        return "{}".format(SpreadNetIn.basename())


