import torch
import torch.nn as nn
import torch.nn.functional as F


class MakeSomeNoiseReal(nn.Module):
    """"""

    def __init__(self, in_ch=1, n_out=256, gaussian_noise=0.5):
        """"""
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 8, 3)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.conv3 = nn.Conv1d(16, 32, 3)
        self.conv4 = nn.Conv1d(32, 64, 3)
        self.conv5 = nn.Conv1d(64, 128, 3)
        self.conv6 = nn.Conv1d(128, 128, 3)
        self.conv7 = nn.Conv1d(128, 128, 3)
        self.conv8 = nn.Conv1d(128, 256, 3)
        self.conv9 = nn.Conv1d(256, 256, 3)
        self.conv10 = nn.Conv1d(256, 256, 3)

        self.bn0 = nn.BatchNorm1d(1)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_out)

        self.gaus_noise = gaussian_noise

    def forward(self, X):
        """"""
        x = self.bn0(X)

        if self.training:
            e = torch.randn(X.shape)
            if X.is_cuda: e = e.cuda()
            x = x + e * self.gaus_noise

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        x = self.pool(F.relu(self.conv8(x)))
        x = self.pool(F.relu(self.bn9(self.conv9(x))))
        x = self.pool(F.relu(self.conv10(x)))

        x = F.dropout(x.view(X.size(0), -1), training=self.training, p=0.5)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training, p=0.5)
        x = self.out(x)

        return x

    @staticmethod
    def save_name(args):
        return "{}".format(MakeSomeNoiseReal.basename())

    @staticmethod
    def basename():
        return MakeSomeNoiseReal.__name__

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = MakeSomeNoiseReal()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return MakeSomeNoiseReal()
