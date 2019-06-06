import torch
import torch.nn as nn

from util import device


class BigChannels(nn.Module):
    def __init__(self, input_shape, out_shape, kernel_size):
        super(BigChannels, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        ############
        # SETTINGS #
        ############
        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = 2

        #################
        # DEFINE BLOCKS #
        #################
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 1024, self.kernel_size, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(1024)
        ).to(device)
        self.block2 = nn.Sequential(
            nn.Conv1d(1024, 256, 3, padding=0),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ).to(device)
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=0),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ).to(device)
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ).to(device)

        ########################
        # CLASSIFICATION BLOCK #
        ########################
        self.classification_block = nn.Sequential(
            nn.Dropout(p=0.5),
            torch.nn.Linear(2176, 256).to(device),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            torch.nn.Linear(256, self.out_shape).to(device)
        ).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        # Start block
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Classification
        x = self.classification_block(x)
        return x

    def name(self):
        return "{}_k{}".format(BigChannels.basename(), self.kernel_size)

    @staticmethod
    def save_name(args):
        return "{}_k{}".format(BigChannels.basename(), args['kernel_size'])

    @staticmethod
    def basename():
        return BigChannels.__name__

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape,
            'kernel_size': self.kernel_size,
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = BigChannels(input_shape=checkpoint['input_shape'],
                            out_shape=checkpoint['out_shape'],
                            kernel_size=checkpoint['kernel_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return BigChannels(out_shape=args['n_classes'],
                           input_shape=args['input_shape'],
                           kernel_size=args['kernel_size'])
