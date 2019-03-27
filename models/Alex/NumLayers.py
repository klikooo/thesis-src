import torch
import torch.nn as nn
import torch.nn.functional as F

from util import device


class NumLayers(nn.Module):
    def __init__(self, input_shape, out_shape, num_layers, kernel_size, channel_size):
        super(NumLayers, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = 2
        self.channel_size = channel_size
        self.num_layers = num_layers

        self.conv_layers = []
        self.bn_layers = []
        self.mp_layers = []

        in_channels = 1
        out_channels = channel_size
        num_features = input_shape

        for i in range(self.num_layers):
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding).to(device))
            self.bn_layers.append(
                nn.BatchNorm1d(num_features=out_channels).to(device))
            self.mp_layers.append(
                nn.MaxPool1d(self.max_pool).to(device))
            num_features = int(num_features / 2)
            in_channels = out_channels
            out_channels = 2 * in_channels

        self.drop_out = nn.Dropout(p=0.5)

        self.fc4 = torch.nn.Linear(int(in_channels * num_features), 300).to(device)
        self.fc5 = torch.nn.Linear(300, 300).to(device)
        self.fc6 = torch.nn.Linear(300, self.out_shape).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        x = inputs
        for i in range(self.num_layers):
                x = self.mp_layers[i](F.relu(
                    self.bn_layers[i](self.conv_layers[i](x))
                ))

        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Perform dropout
        x = self.drop_out(x)

        # Perform MLP
        x = self.fc4(x).to(device)
        x = F.relu(x).to(device)
        x = self.drop_out(x)
        x = self.fc5(x).to(device)
        x = self.drop_out(x)
        x = F.relu(x).to(device)

        # Final layer without ReLU
        x = self.fc6(x).to(device)
        return x

    def name(self):
        return "NumLayers_k{}_c{}_l{}".format(self.kernel_size, self.channel_size, self.num_layers)

    @staticmethod
    def save_name(args):
        return "{}_k{}_c{}_l{}".format(NumLayers.basename(), args['kernel_size'],
                                       args['channel_size'], args['num_layers'])

    @staticmethod
    def basename():
        return "NumLayers"

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape,
            'kernel_size': self.kernel_size,
            'channel_size': self.channel_size,
            'num_layers': self.num_layers
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = NumLayers(input_shape=checkpoint['input_shape'],
                          out_shape=checkpoint['out_shape'],
                          kernel_size=checkpoint['kernel_size'],
                          channel_size=checkpoint['channel_size'],
                          num_layers=checkpoint['num_layers'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return NumLayers(out_shape=args['n_classes'],
                         input_shape=args['input_shape'],
                         kernel_size=args['kernel_size'],
                         num_layers=args['num_layers'],
                         channel_size=args['channel_size'])
