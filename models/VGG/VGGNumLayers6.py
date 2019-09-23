import torch
import torch.nn as nn

from util import device


# Select max pool value, no dropout for classification block, less neurons in classification block, average pooling,
# more dense layers
class VGGNumLayers6(nn.Module):
    def __init__(self, input_shape, out_shape, num_layers, kernel_size, channel_size, max_pool=4):
        super(VGGNumLayers6, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        ############
        # SETTINGS #
        ############
        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = max_pool
        self.channel_size = channel_size
        self.num_layers = num_layers
        self.max_channels = 256
        self.num_blocks = 2

        num_features = input_shape

        #################
        # DEFINE BLOCKS #
        #################
        self.block1 = self.first_conv_block(channel_size, self.num_layers)
        num_features = self.calc_num_features(num_features)

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
            torch.nn.Linear(int(channel_size * num_features), 8).to(device),
            nn.ReLU(),
            torch.nn.Linear(8, 8).to(device),
            nn.ReLU(),
            torch.nn.Linear(8, 8).to(device),
            nn.ReLU(),
            torch.nn.Linear(8, self.out_shape).to(device)
        ).to(device)

    def conv_block(self, in_channels, num_layers):
        out_channels = in_channels * 2
        conv_layers = []
        if out_channels > self.max_channels:
            out_channels = self.max_channels

        for i in range(num_layers):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding))
            conv_layers.append(nn.ReLU())

            in_channels = out_channels

        return nn.Sequential(
            nn.Sequential(*conv_layers),
            nn.AvgPool1d(self.max_pool),
            nn.BatchNorm1d(out_channels)
        ).to(device)

    def first_conv_block(self, out_channels, num_layers):
        conv_layers = []
        in_channels = 1
        if out_channels > self.max_channels:
            out_channels = self.max_channels
        for i in range(num_layers):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding))
            conv_layers.append(nn.ReLU())

            in_channels = out_channels

        return nn.Sequential(
            nn.Sequential(*conv_layers),
            nn.AvgPool1d(self.max_pool),
            nn.BatchNorm1d(out_channels)
        ).to(device)

    def calc_num_features(self, num_features):
        for i in range(self.num_layers):
            num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        return int(num_features / self.max_pool)

    def calc_num_channels(self, in_channels):
        res = int(in_channels * 2)
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
        return "{}_k{}_c{}_l{}_m{}".format(VGGNumLayers6.basename(),
                                           self.kernel_size, self.channel_size, self.num_layers,
                                           self.max_pool)

    @staticmethod
    def save_name(args):
        return "{}_k{}_c{}_l{}_m{}".format(VGGNumLayers6.basename(), args['kernel_size'],
                                           args['channel_size'], args['num_layers'], args['max_pool'])

    @staticmethod
    def basename():
        return VGGNumLayers6.__name__

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape,
            'kernel_size': self.kernel_size,
            'channel_size': self.channel_size,
            'num_layers': self.num_layers,
            'max_pool': self.max_pool
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = VGGNumLayers6(input_shape=checkpoint['input_shape'],
                              out_shape=checkpoint['out_shape'],
                              kernel_size=checkpoint['kernel_size'],
                              channel_size=checkpoint['channel_size'],
                              num_layers=checkpoint['num_layers'],
                              max_pool=checkpoint['max_pool'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return VGGNumLayers6(out_shape=args['n_classes'],
                             input_shape=args['input_shape'],
                             kernel_size=args['kernel_size'],
                             num_layers=args['num_layers'],
                             channel_size=args['channel_size'],
                             max_pool=args['max_pool'])
