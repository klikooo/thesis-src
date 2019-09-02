import torch
import torch.nn as nn

from util import device


class ZaidCNNMasked(nn.Module):
    def __init__(self, input_shape, out_shape):
        super(ZaidCNNMasked, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=1, padding=0),
            nn.SELU(),
            nn.BatchNorm1d(4),
            nn.AvgPool1d(2, stride=2),
        ).to(device)

        self.fc = nn.Sequential(
            nn.Linear(input_shape*2, 10),
            nn.SELU(),
            nn.Linear(10, 10),
            nn.SELU(),
            nn.Linear(10, out_shape)
        ).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        x = self.cnn(inputs)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

    def name(self):
        return ZaidCNNMasked.basename()

    @staticmethod
    def save_name(args):
        return ZaidCNNMasked.basename()

    @staticmethod
    def basename():
        return ZaidCNNMasked.__name__

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape,
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = ZaidCNNMasked(input_shape=checkpoint['input_shape'],
                              out_shape=checkpoint['out_shape'],
                              )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return ZaidCNNMasked(out_shape=args['n_classes'],
                             input_shape=args['input_shape'],
                             )
