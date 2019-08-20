import torch
import torch.nn as nn

from util import device


class SmallCNN(nn.Module):
    def __init__(self, input_shape, out_shape, kernel_size, channel_size=128, max_pool=9):
        super(SmallCNN, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = max_pool
        self.channel_size = channel_size

        self.cnn = nn.Sequential(
            nn.Conv1d(1, self.channel_size, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(self.max_pool),
            nn.BatchNorm1d(self.channel_size),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        ).to(device)

        num_features = input_shape + 2 * self.padding - 1 * (self.kernel_size - 1)
        num_features = int(num_features / self.max_pool)
        num_features = num_features + 2 * 1 - 1 * (3 - 1)
        num_features = int(num_features / 2)

        self.fc = nn.Sequential(
            torch.nn.Linear(num_features * self.channel_size, 256),
            nn.ReLU(),
            torch.nn.Linear(256, 256),
            nn.ReLU(),
            torch.nn.Linear(256, self.out_shape)
        ).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        x = self.cnn(inputs)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

    def name(self):
        return "{}_k{}_m{}_c{}".format(SmallCNN.basename(), self.kernel_size, self.max_pool, self.channel_size)

    @staticmethod
    def save_name(args):
        return "{}_k{}_m{}_c{}".format(SmallCNN.basename(), args['kernel_size'], args['max_pool'], args['channel_size'])

    @staticmethod
    def basename():
        return SmallCNN.__name__

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape,
            'kernel_size': self.kernel_size,
            'channel_size': self.channel_size,
            'max_pool': self.max_pool
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = SmallCNN(input_shape=checkpoint['input_shape'],
                         out_shape=checkpoint['out_shape'],
                         kernel_size=checkpoint['kernel_size'],
                         channel_size=checkpoint['channel_size'],
                         max_pool=checkpoint['max_pool'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return SmallCNN(out_shape=args['n_classes'],
                        input_shape=args['input_shape'],
                        kernel_size=args['kernel_size'],
                        max_pool=args['max_pool'],
                        channel_size=args['channel_size'])
