import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util import device
from scipy.stats import norm
import math


class SpreadV3(nn.Module):
    def __init__(self, spread_factor, input_shape, out_shape):
        super(SpreadV3, self).__init__()

        self.spread_factor = spread_factor
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.spread_neurons = 100 * self.spread_factor

        self.fc1 = nn.Linear(self.input_shape, 100).to(device)
        self.bn = nn.BatchNorm1d(100).to(device)
        self.fc2 = nn.Linear(self.spread_neurons, self.out_shape).to(device)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)

        self.min_constant = torch.tensor(-4.0, requires_grad=False)
        self.max_constant = torch.tensor(4.0, requires_grad=False)

        self.intermediate_values = []

        step = float(1.0 / (self.spread_factor+1))
        threshold = 0.000001
        right = step
        centroids = []
        for i in range(int(spread_factor)):
            r2 = right if right != 1.0 else right - threshold
            x = norm.ppf(r2)
            if math.isnan(x):
                x = centroids[0] * -1.0
            centroids.append(x)

            right += step
        self.centroids = torch.tensor(centroids, requires_grad=False).to(device).unsqueeze(0).to(device)

    def forward(self, x):
        batch_size = x.size()[0]

        x = F.relu(self.fc1(x)).to(device)

        x = self.bn(x)
        min_tensor = self.bn.weight * self.min_constant + self.bn.bias
        max_tensor = self.bn.weight * self.max_constant + self.bn.bias

        x = self.spread(x, min_tensor, max_tensor, batch_size, self.bn.weight, self.bn.bias)
        # self.intermediate_values.append(x.detach().cpu().numpy())

        x = self.fc2(x).to(device)
        return x

    def spread(self, x, tensor_min, tensor_max, batch_size, alpha, beta, num_neurons=100):
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
        centroids = self.centroids  # torch.arange(0.5, self.spread_factor).to(device).unsqueeze(0).to(device)
        # TODO: fix something in contguous, which could make it faster

        # we need to create batch_size x num_neurons centroids vector with [c0, c1,c2,c3,c4,c5, c0,c1,c2 .....] * 100
        # print(centroids.expand(batch_size * num_neurons, -1).size())
        centroids = centroids.expand(batch_size * num_neurons, -1).contiguous().view(num_neurons,
                                                                                     batch_size * self.spread_factor)
        # print(num_neurons)
        # print(self.spread_factor)
        # print(centroids[0])
        # print(centroids.size())
        # exit(0)
        alpha = alpha.expand((num_neurons * self.spread_factor, batch_size))
        beta = beta.expand((num_neurons * self.spread_factor, batch_size))
        centroids = alpha.t() * centroids + beta.t()
        centroids = centroids.float()

        # Calulate the part of function n_c => max(1, 1 - |c - x'|)
        absolute = 1 - (centroids - x_spread).abs()
        nc = torch.max(torch.zeros([1]).to(device).float(), absolute).to(device)

        # Calculate the whole of the nc function and reshape to the correct size
        left_centroid = centroids[:, 0].expand(num_neurons * self.spread_factor, batch_size).t()
        right_centroid = centroids[:, self.spread_factor-1].expand(num_neurons * self.spread_factor, batch_size).t()

        result = torch.where(
            (x_spread < centroids).__and__(centroids < left_centroid).__or__(
                (x_spread > centroids).__and__(centroids < right_centroid)
            ),
            torch.ones([1]).to(device).float(),
            nc.float()
        ).to(device)
        res = result.transpose(0, 1).contiguous().view(batch_size, num_neurons * self.spread_factor)
        return res

    def name(self):
        return SpreadV3.basename()

    @staticmethod
    def basename():
        return SpreadV3.__name__

    @staticmethod
    def save_name(args):
        return "{}".format(SpreadV3.basename())

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

        model = SpreadV3(checkpoint['sf'], input_shape=checkpoint['input_shape'], out_shape=checkpoint['out_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return SpreadV3(spread_factor=args['sf'], out_shape=args['n_classes'], input_shape=args['input_shape'])


