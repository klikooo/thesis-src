import torch
import torch.nn as nn

from util import device


# VGG like networks require many filters


class KernelBigTest(nn.Module):
    def __init__(self, input_shape, out_shape, kernel_size, channel_size):
        super(KernelBigTest, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = 2
        self.channel_size = channel_size

        self.conv1_channels = self.channel_size
        self.conv2_channels = self.conv1_channels * 2
        self.conv3_channels = self.conv2_channels * 2
        self.conv4_channels = self.conv3_channels * 2
        self.conv5_channels = self.conv4_channels * 2
        num_features = input_shape

        # BLOCK 1
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(num_features=1).to(device),
            nn.Conv1d(1, self.conv1_channels, kernel_size=self.kernel_size, padding=self.padding).to(device),
            nn.ReLU(),
            nn.MaxPool1d(self.max_pool).to(device),
            nn.BatchNorm1d(num_features=self.conv1_channels).to(device)
        ).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        num_features = int(num_features / 2)

        # BLOCK 2
        self.block2 = nn.Sequential(
            nn.Conv1d(self.conv1_channels, self.conv2_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv1d(self.conv2_channels, self.conv3_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.conv3_channels).to(device)
        ).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)

        # BLOCK 3
        self.block3 = nn.Sequential(
            nn.Conv1d(self.conv3_channels, self.conv4_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv1d(self.conv4_channels, self.conv5_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.conv5_channels).to(device)
        ).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)

        # CLASSIFICATION BLOCK
        self.classification_block = nn.Sequential(
            nn.Dropout(p=0.5),
            torch.nn.Linear(int(self.conv5_channels * num_features), 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            torch.nn.Linear(256, self.out_shape)
        ).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)

        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Perform dropout
        x = self.classification_block(x)
        return x

    def name(self):
        return "{}_k{}_c{}".format(KernelBigTest.basename(), self.kernel_size, self.channel_size)

    @staticmethod
    def basename():
        return KernelBigTest.__name__

    @staticmethod
    def save_name(args):
        return "{}_k{}_c{}".format(KernelBigTest.basename(), args['kernel_size'],
                                   args['channel_size'])

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape,
            'kernel_size': self.kernel_size,
            'channel_size': self.channel_size
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = KernelBigTest(input_shape=checkpoint['input_shape'],
                              out_shape=checkpoint['out_shape'],
                              kernel_size=checkpoint['kernel_size'],
                              channel_size=checkpoint['channel_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return KernelBigTest(out_shape=args['n_classes'],
                             input_shape=args['input_shape'],
                             kernel_size=args['kernel_size'],
                             channel_size=args['channel_size'])
