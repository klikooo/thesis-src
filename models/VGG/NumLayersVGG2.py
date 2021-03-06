import torch
import torch.nn as nn
import torch.nn.functional as F

from util import device


class NumLayersVGG2(nn.Module):
    def __init__(self, input_shape, out_shape, num_layers, kernel_size, channel_size):
        super(NumLayersVGG2, self).__init__()
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

        self.bn0 = nn.BatchNorm1d(num_features=1).to(device)

        num_features = input_shape

        max_channels = 128

        self.num_blocks = 2

        # Starting point
        self.conv1 = nn.Conv1d(1, channel_size,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.mp1 = nn.MaxPool1d(self.max_pool).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=channel_size).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        num_features = int(num_features / 2)

        in_channels = channel_size
        out_channels = channel_size * 2

        for i in range(self.num_blocks):
            self.conv_layers.append([])
            for j in range(self.num_layers):
                self.conv_layers[i].append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding).to(device))
                num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)

                in_channels = out_channels
                out_channels = 2 * in_channels

                # Keep the number of channels to a certain max
                if out_channels > max_channels:
                    out_channels = max_channels

            # Add batch normalization layers (remember in_channels was the previous out_channels)
            self.bn_layers.append(
                nn.BatchNorm1d(num_features=in_channels).to(device))
            # Cut features depending on the value for max pool
            num_features = int(num_features / self.max_pool)

        # Dropout and pool vars
        self.drop_out = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool1d(self.max_pool).to(device)

        # The fully connected layers
        self.fc1 = torch.nn.Linear(int(in_channels * num_features), 256).to(device)
        self.fc2 = torch.nn.Linear(256, self.out_shape).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        x = self.bn0(inputs)
        x = self.bn1(self.mp1(F.relu(self.conv1(x))))
        for i in range(self.num_blocks):
            for j in range(self.num_layers):
                x = F.relu(self.conv_layers[i][j](x))

            x = self.bn_layers[i](self.pool(x))
        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Perform dropout
        x = self.drop_out(x)

        # Perform MLP
        x = F.relu(self.fc1(x))
        x = self.drop_out(x)
        x = self.fc2(x).to(device)
        return x

    def print(self):

        print(self.bn0)
        print(self.conv1)
        print(self.mp1)
        print(self.bn1)
        # x = self.bn0(inputs)
        # x = self.bn1(self.mp1(F.relu(self.conv1(x))))
        for i in range(self.num_blocks):
            for j in range(self.num_layers):
                print("Relu({})".format(self.conv_layers[i][j]))
                # x = F.relu(self.conv_layers[i][j](x))

            print(self.pool)
            print(self.bn_layers[i])
            # x = self.bn_layers[i](self.pool(x))
        # Reshape data for classification
        # x = x.view(batch_size, -1)

        # Perform dropout
        # x = self.drop_out(x)

        # Perform MLP
        # x = F.relu(self.fc1(x))
        # x = self.drop_out(x)
        # x = self.fc2(x).to(device)
        # return x


    def name(self):
        return "{}_k{}_c{}_l{}".format(NumLayersVGG2.basename(),
                                       self.kernel_size, self.channel_size, self.num_layers)

    @staticmethod
    def save_name(args):
        return "{}_k{}_c{}_l{}".format(NumLayersVGG2.basename(), args['kernel_size'],
                                       args['channel_size'], args['num_layers'])

    @staticmethod
    def basename():
        return NumLayersVGG2.__name__

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

        model = NumLayersVGG2(input_shape=checkpoint['input_shape'],
                              out_shape=checkpoint['out_shape'],
                              kernel_size=checkpoint['kernel_size'],
                              channel_size=checkpoint['channel_size'],
                              num_layers=checkpoint['num_layers'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return NumLayersVGG2(out_shape=args['n_classes'],
                             input_shape=args['input_shape'],
                             kernel_size=args['kernel_size'],
                             num_layers=args['num_layers'],
                             channel_size=args['channel_size'])
