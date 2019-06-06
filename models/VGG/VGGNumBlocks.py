import torch
import torch.nn as nn

from util import device


class VGGNumBlocks(nn.Module):
    def __init__(self, input_shape, out_shape, num_layers, kernel_size, channel_size):
        super(VGGNumBlocks, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        ############
        # SETTINGS #
        ############
        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = 2
        self.channel_size = channel_size
        self.max_channels = 512
        self.num_layers = num_layers

        self.conv_layers_in_block = 2

        ###################
        # GENERATE BLOCKS #
        ###################
        in_channels = 1
        out_channels = self.channel_size
        blocks = []
        for i in range(self.num_layers):
            block, out_channels = self.conv_block(in_channels, out_channels)
            blocks.append(block)
            in_channels = out_channels
            out_channels = out_channels * 2
        out_channels = out_channels / 2
        self.blocks = nn.Sequential(*blocks).to(device)
        num_features = self.calc_num_features(input_shape)

        ########################
        # CLASSIFICATION BLOCK #
        ########################
        self.classification_block = nn.Sequential(
            # nn.Dropout(p=0.5),
            torch.nn.Linear(int(out_channels * num_features), 4000).to(device),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            torch.nn.Linear(4000, self.out_shape).to(device)
        ).to(device)

    def conv_block(self, in_channels, out_channels):
        if out_channels >= self.max_channels:
            out_channels = self.max_channels
        block = [nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
                 # nn.ReLU(),
                 # nn.Conv1d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
                 nn.ReLU(),
                 nn.AvgPool1d(self.max_pool),
                 nn.BatchNorm1d(out_channels)]
        if out_channels >= self.max_channels:
            out_channels = self.max_channels
        return nn.Sequential(*block).to(device), out_channels

    def calc_num_features(self, num_features):
        for i in range(self.num_layers):
            num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
            # num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
            num_features = int(num_features / self.max_pool)
        return num_features

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        # Perform CNN part
        x = self.blocks(inputs)

        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Classification
        x = self.classification_block(x)
        return x

    def name(self):
        return "{}_k{}_c{}_b{}".format(VGGNumBlocks.basename(),
                                       self.kernel_size, self.channel_size, self.num_layers)

    @staticmethod
    def save_name(args):
        return "{}_k{}_c{}_b{}".format(VGGNumBlocks.basename(), args['kernel_size'],
                                       args['channel_size'], args['num_layers'])

    @staticmethod
    def basename():
        return VGGNumBlocks.__name__

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

        model = VGGNumBlocks(input_shape=checkpoint['input_shape'],
                             out_shape=checkpoint['out_shape'],
                             kernel_size=checkpoint['kernel_size'],
                             channel_size=checkpoint['channel_size'],
                             num_layers=checkpoint['num_layers'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return VGGNumBlocks(out_shape=args['n_classes'],
                            input_shape=args['input_shape'],
                            kernel_size=args['kernel_size'],
                            num_layers=args['num_layers'],
                            channel_size=args['channel_size'])
