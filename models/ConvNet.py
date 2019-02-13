import torch
import torch.nn as nn
import torch.nn.functional as F


from util import device


class ConvNet(nn.Module):
    def __init__(self, input_shape, out_shape):
        super(ConvNet, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.hidden_size = 100

        self.conv1 = nn.Conv1d(1, 32, kernel_size=11, padding=3).to(device)
        self.mp1 = nn.MaxPool1d(5).to(device)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=3).to(device)
        self.mp2 = nn.MaxPool1d(5).to(device)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=11, padding=3).to(device)
        self.mp3 = nn.MaxPool1d(5).to(device)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=11, padding=3).to(device)
        self.mp4 = nn.MaxPool1d(5).to(device)

        # self.features = torch.nn.Sequential(
        #     nn.Conv1d(self.input_shape, 50, kernel_size=2, padding=1).to(device),
        #     nn.ReLU(),
        #     nn.Conv1d(self.input_shape, 50, kernel_size=2, padding=2).to(device),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2).to(device),
        #     nn.Conv1d(self.input_shape, 100, kernel_size=2, padding=1).to(device),
        #     nn.ReLU(),
        #     nn.Conv1d(self.input_shape, 100, kernel_size=2, padding=2).to(device),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2).to(device),
        # )

        self.fc4 = torch.nn.Linear(512, 400).to(device)
        self.fc5 = torch.nn.Linear(400, 400).to(device)
        self.fc6 = torch.nn.Linear(400, self.out_shape).to(device)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # print('Inputs original size {}'.format(x.size()))
        batch_size = x.size()[0]
        # print('Batch size: {}'.format(batch_size))
        # exit()
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()
        # print('Inputs size: {}'.format(inputs.size()))

        # print('Inputs size {}'.format(inputs.size()))
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.mp3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.mp4(x)

        # x = self.features(inputs)
        # exit()
        # print('Shape x {}'.format(x.size()))
        x = x.view(batch_size, -1)
        # print('Shape x {}'.format(x.size()))
        # exit()

        # print('Conv1 size: {}'.format(test.size()))
        # exit()

        # x = self.features(inputs)

        x = self.fc4(x).to(device)
        x = F.relu(x).to(device)
        x = self.fc5(x).to(device)
        x = F.relu(x).to(device)

        x = self.fc6(x).to(device)
        return x

    def name(self):
        return "ConvNet"

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = ConvNet(input_shape=checkpoint['input_shape'], out_shape=checkpoint['out_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.hidden_size = 100
        return model

    @staticmethod
    def init(args):
        return ConvNet(out_shape=args['n_classes'], input_shape=args['input_shape'])
