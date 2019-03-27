import torch
import torch.nn as nn
import torch.nn.functional as F


from util import device


class ConvNetDPA(nn.Module):
    def __init__(self, input_shape, out_shape):
        super(ConvNetDPA, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.conv1 = nn.Conv1d(1, 32, kernel_size=11, padding=3).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=32).to(device)
        self.mp1 = nn.MaxPool1d(5).to(device)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=3).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=64).to(device)
        self.mp2 = nn.MaxPool1d(5).to(device)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=11, padding=3).to(device)
        self.bn3 = nn.BatchNorm1d(num_features=128).to(device)
        self.mp3 = nn.MaxPool1d(5).to(device)

        self.fc4 = torch.nn.Linear(512+256, 400).to(device)
        self.fc5 = torch.nn.Linear(400, 400).to(device)
        self.fc6 = torch.nn.Linear(400, self.out_shape).to(device)

    def forward(self, x, plaintext):
        # Reshape input
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        # Perform convolution
        x = self.mp1(F.relu(self.bn1(self.conv1(inputs))))
        x = self.mp2(F.relu(self.bn2(self.conv2(x))))
        x = self.mp3(F.relu(self.bn3(self.conv3(x))))

        # Reshape data for classification and add the plaintext
        x = x.view(batch_size, -1)
        # print('x: {}'.format(x.size()))
        # print('plain {}'.format(plaintext.size()))
        x = torch.cat([plaintext.float(), x], 1)

        # Perform MLP
        x = self.fc4(x).to(device)
        x = F.relu(x).to(device)
        x = self.fc5(x).to(device)
        x = F.relu(x).to(device)

        # Final layer without ReLU
        x = self.fc6(x).to(device)
        return x

    def name(self):
        return "ConvNetDPA"

    @staticmethod
    def filename():
        return "ConvNetDPA"

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = ConvNetDPA(input_shape=checkpoint['input_shape'], out_shape=checkpoint['out_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return ConvNetDPA(out_shape=args['n_classes'], input_shape=args['input_shape'])
