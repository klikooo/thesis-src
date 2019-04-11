import torch
import torch.nn as nn
import torch.nn.functional as F

from util import device


class NumLayersVGG(nn.Module):
    def __init__(self, input_shape, out_shape, num_layers, kernel_size, channel_size):
        super(NumLayersVGG, self).__init__()
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

        self.num_blocks = 3

        for j in range(self.num_blocks):
            self.conv_layers.append([])
            for i in range(self.num_layers):
                self.conv_layers[j].append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding).to(device))
                num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
                in_channels = out_channels

            self.bn_layers.append(
                nn.BatchNorm1d(num_features=out_channels).to(device))
            num_features = int(num_features / 2)
            in_channels = out_channels
            out_channels = 2 * in_channels

        self.drop_out = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool1d(self.max_pool).to(device)

        self.fc1 = torch.nn.Linear(int(in_channels * num_features), 256).to(device)
        self.fc2 = torch.nn.Linear(256, self.out_shape).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        x = inputs
        for j in range(self.num_blocks):
            for i in range(self.num_layers):
                x = self.conv_layers[j][i](x)

            # TODO: should I do BN before pooling or after?
            x = self.pool(F.relu(x))
            x = self.bn_layers[j](x)
        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Perform dropout
        x = self.drop_out(x)

        # Perform MLP
        x = F.relu(self.fc1(x).to(device)).to(device)
        x = self.drop_out(x)
        x = self.fc2(x).to(device).to(device)
        return x

    def name(self):
        return "{}_k{}_c{}_l{}".format(NumLayersVGG.basename(),
                                       self.kernel_size, self.channel_size, self.num_layers)

    @staticmethod
    def save_name(args):
        return "{}_k{}_c{}_l{}".format(NumLayersVGG.basename(), args['kernel_size'],
                                       args['channel_size'], args['num_layers'])

    @staticmethod
    def basename():
        return NumLayersVGG.__name__

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

        model = NumLayersVGG(input_shape=checkpoint['input_shape'],
                             out_shape=checkpoint['out_shape'],
                             kernel_size=checkpoint['kernel_size'],
                             channel_size=checkpoint['channel_size'],
                             num_layers=checkpoint['num_layers'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return NumLayersVGG(out_shape=args['n_classes'],
                            input_shape=args['input_shape'],
                            kernel_size=args['kernel_size'],
                            num_layers=args['num_layers'],
                            channel_size=args['channel_size'])
