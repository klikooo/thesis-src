import torch
import torch.nn as nn
import torch.nn.functional as F

from util import device


class ConvNetKernelSmall(nn.Module):
    def __init__(self, input_shape, out_shape, kernel_size, channel_size=8):
        super(ConvNetKernelSmall, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = 2

        self.conv1_channels = channel_size
        self.conv1 = nn.Conv1d(1, self.conv1_channels, kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=self.conv1_channels).to(device)
        self.mp1 = nn.MaxPool1d(self.max_pool).to(device)
        size = int(input_shape * self.conv1_channels / self.max_pool)

        self.conv2_channels = self.conv1_channels * 2
        self.conv2 = nn.Conv1d(self.conv1_channels, self.conv2_channels, kernel_size=3,
                               padding=1).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=self.conv2_channels).to(device)
        self.mp2 = nn.MaxPool1d(5).to(device)
        size = int(size * 2 / 5)

        self.conv3_channels = self.conv2_channels * 2
        self.conv3 = nn.Conv1d(self.conv2_channels, self.conv3_channels, kernel_size=3,
                               padding=1).to(device)
        self.bn3 = nn.BatchNorm1d(num_features=self.conv3_channels).to(device)
        self.mp3 = nn.MaxPool1d(5).to(device)
        size = int(size * 2 / 5)

        self.conv4 = nn.Conv1d(self.conv3_channels, self.conv3_channels, kernel_size=3,
                               padding=1).to(device)
        self.bn4 = nn.BatchNorm1d(num_features=self.conv3_channels).to(device)
        self.mp4 = nn.MaxPool1d(5).to(device)

        self.drop_out = nn.Dropout(p=0.5)

        # self.drop1 = nn.Dropout(p=0.03)
        size = int(size / 5)

        # self.conv5 = nn.Conv1d(128, 128, kernel_size=16, padding=8).to(device)
        # self.bn5 = nn.BatchNorm1d(num_features=128).to(device)
        # self.mp5 = nn.MaxPool1d(4).to(device)

        self.fc4 = torch.nn.Linear(int(size), 200).to(device)
        self.fc5 = torch.nn.Linear(200, 200).to(device)
        self.fc6 = torch.nn.Linear(200, self.out_shape).to(device)

    def forward(self, x):
        # print('Inputs original size {}'.format(x.size()))
        batch_size = x.size()[0]
        # print('Batch size: {}'.format(batch_size))
        # exit()
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()
        # print('Inputs size: {}'.format(inputs.size()))

        # print('Inputs size {}'.format(inputs.size()))
        x = self.mp1(F.relu(self.bn1(self.conv1(inputs))))
        x = self.mp2(F.relu(self.bn2(self.conv2(x))))
        x = self.mp3(F.relu(self.bn3(self.conv3(x))))
        x = self.mp4(F.relu(self.bn4(self.conv4(x))))

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
        return "{}_k{}".format(ConvNetKernelSmall.basename(), self.kernel_size)

    @staticmethod
    def save_name(args):
        return "{}_k{}".format(ConvNetKernelSmall.basename(), args['kernel_size'])

    @staticmethod
    def basename():
        return "ConvNetKernelSmall"

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

        model = ConvNetKernelSmall(input_shape=checkpoint['input_shape'],
                                   out_shape=checkpoint['out_shape'],
                                   kernel_size=checkpoint['kernel_size'],
                                   channel_size=checkpoint['channel_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return ConvNetKernelSmall(out_shape=args['n_classes'],
                                  input_shape=args['input_shape'],
                                  kernel_size=args['kernel_size'])
