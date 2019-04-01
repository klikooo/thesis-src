import torch
import torch.nn as nn
import torch.nn.functional as F

from util import device


class KBS2(nn.Module):
    def __init__(self, input_shape, out_shape, kernel_size, channel_size):
        super(KBS2, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.kernel_size = kernel_size
        self.padding = int(self.kernel_size / 2)
        self.max_pool = 5
        self.channel_size = channel_size

        self.conv1_channels = self.channel_size
        self.conv2_channels = self.conv1_channels * 2

        self.conv1 = nn.Conv1d(1, self.conv1_channels, kernel_size=self.kernel_size, padding=self.padding).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=self.conv1_channels).to(device)
        self.mp1 = nn.MaxPool1d(self.max_pool).to(device)
        num_features = int(input_shape / self.max_pool)

        self.conv2 = nn.Conv1d(self.conv1_channels, self.conv2_channels, kernel_size=3,
                               padding=1).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=self.conv2_channels).to(device)
        self.mp2 = nn.MaxPool1d(self.max_pool).to(device)
        num_features = int(num_features / self.max_pool)

        self.drop_out = nn.Dropout(p=0.5)
        self.drop_out2 = nn.Dropout(p=0.5)

        self.fc4 = torch.nn.Linear(int(self.conv2_channels * num_features), 300).to(device)
        self.fc5 = torch.nn.Linear(300, self.out_shape).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        x = self.mp1(self.bn1(F.relu(self.conv1(inputs))))
        x = self.mp2(self.bn2(F.relu(self.conv2(x))))

        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Perform dropout
        x = self.drop_out(x)

        # Perform MLP
        # Perform MLP
        x = F.relu(self.fc4(x).to(device)).to(device)
        x = self.drop_out2(x)

        # Final layer without ReLU
        x = self.fc5(x).to(device)
        return x

    def name(self):
        return "{}_k{}_c{}".format(KBS2.basename(), self.kernel_size, self.channel_size)

    @staticmethod
    def save_name(args):
        return "{}_k{}_c{}".format(KBS2.basename(), args['kernel_size'], args['channel_size'])

    @staticmethod
    def basename():
        return "KBS2"

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

        model = KBS2(input_shape=checkpoint['input_shape'],
                     out_shape=checkpoint['out_shape'],
                     kernel_size=checkpoint['kernel_size'],
                     channel_size=checkpoint['channel_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return KBS2(out_shape=args['n_classes'],
                    input_shape=args['input_shape'],
                    kernel_size=args['kernel_size'],
                    channel_size=args['channel_size'])
