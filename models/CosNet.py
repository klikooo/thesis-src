import torch
import torch.nn as nn


from util import device


class CosNet(nn.Module):
    def __init__(self, spread_factor, input_shape, out_shape):
        super(CosNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, 100 * spread_factor).to(device)
        self.fc2 = nn.Linear(100 * spread_factor, 100 * spread_factor).to(device)
        self.spread_factor = spread_factor
        self.out_shape = out_shape
        self.input_shape = input_shape
        self.network_name = "CosNet"

        self.fc3 = nn.Linear(100 * spread_factor, out_shape).to(device)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.cos(self.fc1(x)).to(device)
        x = torch.cos(self.fc2(x)).to(device)

        x = self.fc3(x).to(device)
        return x

    def name(self):
        return self.network_name

    @staticmethod
    def basename():
        return "CosNet"

    @staticmethod
    def save_name(args):
        return "{}".format(CosNet.basename())

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

        print(checkpoint)
        model = CosNet(checkpoint['sf'], input_shape=checkpoint['input_shape'], out_shape=checkpoint['out_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return CosNet(spread_factor=args['sf'], out_shape=args['n_classes'], input_shape=args['input_shape'])
