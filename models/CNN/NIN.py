import torch
import torch.nn as nn
import torch.nn.functional as F


from util import device


class NIN(nn.Module):
    def __init__(self, input_shape, out_shape):
        super(NIN, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.hidden_size = 100

        self.kernel_size = 64
        self.padding = int(self.kernel_size / 2)
        self.max_pool = 2

        self.c1 = self.mlpconv(1, 16, self.kernel_size)
        self.c2 = self.mlpconv(16, 32, self.kernel_size)
        self.c3 = self.mlpconv(32, 64, self.kernel_size)
        self.c4 = self.mlpconv(64, 128, self.kernel_size)
        self.c5 = self.mlpconv(128, 128, self.kernel_size)

        size = int(input_shape/25)
        print('Size: {}'.format(size))

        self.fc4 = torch.nn.Linear(128, 500).to(device)
        self.fc5 = torch.nn.Linear(500, 500).to(device)
        self.fc6 = torch.nn.Linear(500, self.out_shape).to(device)

        self.drop_out = torch.nn.Dropout(p=0.5)

    def mlpconv(self, in_channels, out_channels, kernel_size):
        padding = int(kernel_size / 2)
        layers = []
        layers += [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0)]
        layers += [nn.BatchNorm1d(out_channels)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool1d(5)]
        return nn.Sequential(*layers).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        # Perform convolution
        # print('input shape {}'.format(inputs.size()))
        x = self.c1(inputs)
        # print('Out shape {}'.format(x.size()))
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)

        # Reshape data for classification
        x = x.view(batch_size, -1)
        x = self.drop_out(x)

        # Perform MLP
        x = self.fc4(x).to(device)
        x = F.relu(x).to(device)
        x = self.drop_out(x)

        x = self.fc5(x).to(device)
        x = F.relu(x).to(device)

        # Final layer without ReLU
        x = self.fc6(x).to(device)
        return x

    def name(self):
        return "NIN"

    @staticmethod
    def filename():
        return "NIN"

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = NIN(input_shape=checkpoint['input_shape'], out_shape=checkpoint['out_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.hidden_size = 100
        return model

    @staticmethod
    def init(args):
        return NIN(out_shape=args['n_classes'], input_shape=args['input_shape'])
