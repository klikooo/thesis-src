import math
import torch
import torch.nn as nn

from util import device


class NumLayersVGG3(nn.Module):
    def __init__(self, input_shape, out_shape, num_layers, kernel_size, channel_size):
        super(NumLayersVGG3, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        ############
        # SETTINGS #
        ############
        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = 2
        self.channel_size = channel_size
        self.num_layers = num_layers
        self.max_channels = 128
        self.num_blocks = 2

        num_features = input_shape

        #################
        # DEFINE BLOCKS #
        #################
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(num_features=1).to(device),
            nn.Conv1d(1, channel_size, kernel_size=self.kernel_size, padding=self.padding).to(device),
            nn.ReLU(),
            nn.MaxPool1d(self.max_pool).to(device),
            nn.BatchNorm1d(num_features=channel_size).to(device)
        ).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        num_features = int(num_features / self.max_pool)

        self.block2 = self.conv_block(channel_size, self.num_layers)
        num_features = self.calc_num_features(num_features)
        channel_size = self.calc_num_channels(channel_size)

        self.block3 = self.conv_block(channel_size, self.num_layers)
        num_features = self.calc_num_features(num_features)
        channel_size = self.calc_num_channels(channel_size)

        ########################
        # CLASSIFICATION BLOCK #
        ########################
        self.classification_block = nn.Sequential(
            nn.Dropout(p=0.5),
            torch.nn.Linear(int(channel_size * num_features), 256).to(device),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            torch.nn.Linear(256, self.out_shape).to(device)
        ).to(device)

    def conv_block(self, in_channels, num_layers):
        out_channels = in_channels * 2
        conv_layers = []
        for i in range(num_layers):
            if out_channels > 128:
                out_channels = 128
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding))
            conv_layers.append(nn.ReLU())

            in_channels = out_channels
            out_channels = in_channels * 2

        return nn.Sequential(
            nn.Sequential(*conv_layers),
            nn.MaxPool1d(self.max_pool),
            nn.BatchNorm1d(in_channels)
        ).to(device)

    def calc_num_features(self, num_features):
        for i in range(self.num_layers):
            num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        return int(num_features / self.max_pool)

    def calc_num_channels(self, in_channels,):
        res = int(in_channels * math.pow(2, self.num_layers))
        return self.max_channels if res > self.max_channels else res

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        # Start block
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)

        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Classification
        x = self.classification_block(x)
        return x

    def name(self):
        return "{}_k{}_c{}_l{}".format(NumLayersVGG3.basename(),
                                       self.kernel_size, self.channel_size, self.num_layers)

    @staticmethod
    def save_name(args):
        return "{}_k{}_c{}_l{}".format(NumLayersVGG3.basename(), args['kernel_size'],
                                       args['channel_size'], args['num_layers'])

    @staticmethod
    def basename():
        return NumLayersVGG3.__name__

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

        model = NumLayersVGG3(input_shape=checkpoint['input_shape'],
                              out_shape=checkpoint['out_shape'],
                              kernel_size=checkpoint['kernel_size'],
                              channel_size=checkpoint['channel_size'],
                              num_layers=checkpoint['num_layers'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return NumLayersVGG3(out_shape=args['n_classes'],
                             input_shape=args['input_shape'],
                             kernel_size=args['kernel_size'],
                             num_layers=args['num_layers'],
                             channel_size=args['channel_size'])
