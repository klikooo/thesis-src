import torch
import torch.nn as nn
import torch.nn.functional as F

from util import device


# VGG like networks require many filters


class MakeSomeNoise(nn.Module):
    def __init__(self, input_shape, out_shape, kernel_size, channel_size):
        super(MakeSomeNoise, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = 2
        self.channel_size = channel_size

        self.conv1_channels = self.channel_size  # 8
        self.conv2_channels = self.conv1_channels * 2  # 16
        self.conv3_channels = self.conv2_channels * 2  # 32
        self.conv4_channels = self.conv3_channels * 2  # 64
        self.conv5_channels = self.conv4_channels * 2  # 128
        self.conv6_channels = self.conv5_channels  # 128
        self.conv7_channels = self.conv6_channels  # 128
        self.conv8_channels = self.conv7_channels * 2  # 256
        self.conv9_channels = self.conv8_channels  # 256
        self.conv10_channels = self.conv9_channels  # 256
        num_features = input_shape

        # First BN
        self.bn0 = nn.BatchNorm1d(num_features=1).to(device)

        # Convolutions + BN + MP
        self.conv1 = nn.Conv1d(1, self.conv1_channels,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.conv2 = nn.Conv1d(self.conv1_channels, self.conv2_channels,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.conv3 = nn.Conv1d(self.conv2_channels, self.conv3_channels,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.conv4 = nn.Conv1d(self.conv3_channels, self.conv4_channels,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.conv5 = nn.Conv1d(self.conv4_channels, self.conv5_channels,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.conv6 = nn.Conv1d(self.conv5_channels, self.conv6_channels,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.conv7 = nn.Conv1d(self.conv6_channels, self.conv7_channels,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.conv8 = nn.Conv1d(self.conv7_channels, self.conv8_channels,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.conv9 = nn.Conv1d(self.conv8_channels, self.conv9_channels,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.conv10 = nn.Conv1d(self.conv9_channels, self.conv10_channels,
                                kernel_size=self.kernel_size, padding=self.padding).to(device)

        self.bn1 = nn.BatchNorm1d(num_features=self.conv1_channels).to(device)
        self.bn3 = nn.BatchNorm1d(num_features=self.conv3_channels).to(device)
        self.bn5 = nn.BatchNorm1d(num_features=self.conv5_channels).to(device)
        self.bn7 = nn.BatchNorm1d(num_features=self.conv7_channels).to(device)
        self.bn9 = nn.BatchNorm1d(num_features=self.conv9_channels).to(device)
        self.mp1 = nn.MaxPool1d(self.max_pool).to(device)
        num_features = int(num_features / 2)

        # Dropout
        self.drop_out = nn.Dropout(p=0.5)

        self.fc4 = torch.nn.Linear(int(self.conv10_channels * num_features), 256).to(device)
        self.fc5 = torch.nn.Linear(256, 256).to(device)
        self.fc6 = torch.nn.Linear(256, self.out_shape).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        x = self.bn0(inputs)
        # x = inputs
        x = self.bn1(self.mp1(F.relu(self.conv1(x))))
        x = self.bn3(F.relu(self.conv3(F.relu(self.conv2(x)))))
        x = self.bn5(F.relu(self.conv5(F.relu(self.conv4(x)))))
        x = self.bn7(F.relu(self.conv7(F.relu(self.conv6(x)))))
        x = self.bn9(F.relu(self.conv9(F.relu(self.conv8(x)))))
        x = F.relu(self.conv10(x))

        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Perform dropout
        x = self.drop_out(x)

        # Perform MLP
        x = F.relu(self.fc4(x)).to(device)
        x = F.relu(self.fc5(x)).to(device)

        # Final layer without ReLU
        x = self.fc6(x).to(device)
        return x

    def name(self):
        return "{}_k{}_c{}".format(MakeSomeNoise.basename(), self.kernel_size, self.channel_size)

    @staticmethod
    def basename():
        return MakeSomeNoise.__name__

    @staticmethod
    def save_name(args):
        return "{}_k{}_c{}".format(MakeSomeNoise.basename(), args['kernel_size'],
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

        model = MakeSomeNoise(input_shape=checkpoint['input_shape'],
                              out_shape=checkpoint['out_shape'],
                              kernel_size=checkpoint['kernel_size'],
                              channel_size=checkpoint['channel_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return MakeSomeNoise(out_shape=args['n_classes'],
                             input_shape=args['input_shape'],
                             kernel_size=args['kernel_size'],
                             channel_size=args['channel_size'])
