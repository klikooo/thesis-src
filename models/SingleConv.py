import torch
import torch.nn as nn
import torch.nn.functional as F

from util import device


class SingleConv(nn.Module):
    def __init__(self, input_shape, out_shape, kernel_size, channel_size):
        super(SingleConv, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = 2
        num_features = input_shape

        self.conv1_channels = channel_size
        self.conv1 = nn.Conv1d(1, self.conv1_channels, kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=self.conv1_channels).to(device)
        self.mp1 = nn.MaxPool1d(self.max_pool).to(device)
        num_features = int(num_features/ self.max_pool)

        self.fc4 = torch.nn.Linear(int(self.conv1_channels * num_features), 300).to(device)
        self.fc5 = torch.nn.Linear(300, 300).to(device)
        self.fc6 = torch.nn.Linear(300, self.out_shape).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        x = self.mp1(F.relu(self.bn1(self.conv1(inputs))))

        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Perform dropout
        x = F.relu(self.fc4(x).to(device)).to(device)
        x = F.relu(self.fc5(x).to(device)).to(device)

        # Final layer without ReLU
        x = self.fc6(x).to(device)
        return x

    def name(self):
        return "{}_k{}".format(SingleConv.basename(), self.kernel_size)

    @staticmethod
    def save_name(args):
        return "{}_k{}".format(SingleConv.basename(), args['kernel_size'])

    @staticmethod
    def basename():
        return SingleConv.__name__

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape,
            'kernel_size': self.kernel_size,
            'channel_size': self.conv1_channels
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = SingleConv(input_shape=checkpoint['input_shape'],
                           out_shape=checkpoint['out_shape'],
                           kernel_size=checkpoint['kernel_size'],
                           channel_size=checkpoint['channel_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return SingleConv(out_shape=args['n_classes'],
                          input_shape=args['input_shape'],
                          kernel_size=args['kernel_size'],
                          channel_size=args['channel_size'])
