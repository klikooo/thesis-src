import torch
import torch.nn as nn
import torch.nn.functional as F

from util import device

# VGG like networks require many filters

"""
    VGG format:
        BN
        CNN
        CNN
        BN
        MAX
        CNN
        CNN
        BN
        CNN
        CNN
        BNN
        Dropout
        MLP
"""


class KernelBigVGGDK(nn.Module):
    def __init__(self, input_shape, out_shape, kernel_size, channel_size):
        super(KernelBigVGGDK, self).__init__()
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

        ##########
        # STEP 1 #
        ##########
        self.bn0 = nn.BatchNorm1d(num_features=1).to(device)

        # Convolutions + BN + MP
        self.conv1 = nn.Conv1d(1, self.conv1_channels,
                               kernel_size=self.kernel_size, padding=self.padding).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        self.mp1 = nn.MaxPool1d(self.max_pool).to(device)
        num_features = int(num_features / 2)
        self.bn1 = nn.BatchNorm1d(num_features=self.conv1_channels).to(device)

        ##########
        # STEP 2 #
        ##########
        self.conv2_1 = nn.Conv1d(self.conv1_channels, self.conv2_channels,
                                 kernel_size=self.kernel_size, padding=self.padding).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        self.conv2_2 = nn.Conv1d(self.conv2_channels, self.conv3_channels,
                                 kernel_size=self.kernel_size, padding=self.padding).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        self.mp2 = nn.MaxPool1d(self.max_pool).to(device)
        num_features = int(num_features / 2)
        self.bn2 = nn.BatchNorm1d(num_features=self.conv3_channels).to(device)

        ##########
        # STEP 3 #
        ##########
        self.conv3_1 = nn.Conv1d(self.conv3_channels, self.conv4_channels,
                                 kernel_size=self.kernel_size, padding=self.padding).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        self.conv3_2 = nn.Conv1d(self.conv4_channels, self.conv5_channels,
                                 kernel_size=self.kernel_size, padding=self.padding).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        self.mp3 = nn.MaxPool1d(self.max_pool).to(device)
        num_features = int(num_features / 2)
        self.bn3 = nn.BatchNorm1d(num_features=self.conv5_channels).to(device)

        ##########
        # STEP 4 #
        ##########
        self.conv4_1 = nn.Conv1d(self.conv5_channels, self.conv5_channels,
                                 kernel_size=self.kernel_size, padding=self.padding).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        self.conv4_2 = nn.Conv1d(self.conv5_channels, self.conv5_channels,
                                 kernel_size=self.kernel_size, padding=self.padding).to(device)
        num_features = num_features + 2 * self.padding - 1 * (self.kernel_size - 1)
        self.mp4 = nn.MaxPool1d(self.max_pool).to(device)
        num_features = int(num_features / 2)
        self.bn4 = nn.BatchNorm1d(num_features=self.conv5_channels).to(device)

        ###########
        # Dropout #
        ###########
        self.drop_out = nn.Dropout(p=0.5)
        self.drop_out2 = nn.Dropout(p=0.5)

        ###################
        # FULLY CONNECTED #
        ###################
        self.fc4 = torch.nn.Linear(int(self.conv5_channels * num_features) + 256, 256).to(device)
        self.fc5 = torch.nn.Linear(256, self.out_shape).to(device)

    def forward(self, x, domain_knowledge):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        x = self.bn0(inputs)

        x = self.bn1(self.mp1(F.relu(self.conv1(x))))
        x = self.bn2(self.mp2(F.relu(self.conv2_2(F.relu(self.conv2_1(x))))))
        x = self.bn3(self.mp3(F.relu(self.conv3_2(F.relu(self.conv3_1(x))))))
        x = self.bn4(self.mp4(F.relu(self.conv4_2(F.relu(self.conv4_1(x))))))

        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Add domain knowledge to the features
        x = torch.cat([domain_knowledge.float(), x], 1)

        # Perform dropout
        x = self.drop_out(x)

        # Perform MLP
        x = F.relu(self.fc4(x)).to(device)
        x = self.drop_out2(x)

        # Final layer without ReLU
        x = self.fc5(x).to(device)
        return x

    def name(self):
        return "{}_k{}_c{}".format(KernelBigVGGDK.basename(), self.kernel_size, self.channel_size)

    @staticmethod
    def basename():
        return KernelBigVGGDK.__name__

    @staticmethod
    def save_name(args):
        return "{}_k{}_c{}".format(KernelBigVGGDK.basename(), args['kernel_size'],
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

        model = KernelBigVGGDK(input_shape=checkpoint['input_shape'],
                               out_shape=checkpoint['out_shape'],
                               kernel_size=checkpoint['kernel_size'],
                               channel_size=checkpoint['channel_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return KernelBigVGGDK(out_shape=args['n_classes'],
                              input_shape=args['input_shape'],
                              kernel_size=args['kernel_size'],
                              channel_size=args['channel_size'])
