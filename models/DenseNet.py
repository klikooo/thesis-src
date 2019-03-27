import torch
import torch.nn as nn
import torch.nn.functional as F

from util import device


class DenseNet(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(DenseNet, self).__init__()
        n_hidden = 200
        self.input_shape = input_shape
        self.out_shape = n_classes

        self.fc1 = nn.Linear(input_shape, n_hidden).to(device)
        self.fc2 = nn.Linear(n_hidden, n_hidden).to(device)
        self.fc3 = nn.Linear(n_hidden, n_hidden).to(device)
        self.fc4 = nn.Linear(n_hidden, n_hidden).to(device)
        self.fc5 = nn.Linear(n_hidden, n_hidden).to(device)
        self.fc6 = nn.Linear(n_hidden, n_classes).to(device)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x)).to(device)
        x = F.relu(self.fc2(x)).to(device)
        x = F.relu(self.fc3(x)).to(device)
        x = F.relu(self.fc4(x)).to(device)
        x = F.relu(self.fc5(x)).to(device)

        x = self.fc6(x).to(device)
        # return F.softmax(x, dim=1).to(device)
        return x

    def name(self):
        return "DenseNet"

    @staticmethod
    def basename():
        return "DenseNet"

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = DenseNet(input_shape=checkpoint['input_shape'], n_classes=checkpoint['out_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return DenseNet(n_classes=args['n_classes'], input_shape=args['input_shape'])

    def name(self):
        return "MLPBEST"
