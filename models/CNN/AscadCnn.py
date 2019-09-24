import torch
import torch.nn as nn

from util import device


class AscadCnn(nn.Module):
    def __init__(self, input_shape, out_shape):
        super(AscadCnn, self).__init__()
        self.out_shape = out_shape
        self.input_shape = input_shape

        ############
        # SETTINGS #
        ############
        self.n_classes = out_shape

        padding = 5
        self.cnn_block = nn.Sequential(
            nn.Conv1d(1, 64, 11, stride=2, padding=padding),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, 11, stride=1, padding=padding),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, 11, stride=1, padding=padding),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 512, 11, stride=1, padding=padding),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(512, 512, 11, stride=1, padding=padding),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        ).to(device)

        ########################
        # CLASSIFICATION BLOCK #
        ########################
        self.classification_block = nn.Sequential(
            # torch.nn.Linear(10752, 4096).to(device), # For Ascad Keys
            torch.nn.Linear(5120, 4096).to(device),
            nn.ReLU(),
            torch.nn.Linear(4096, 4096).to(device),
            nn.ReLU(),
            torch.nn.Linear(4096, self.n_classes).to(device),
        ).to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        inputs = x.to(device).view(batch_size, 1, self.input_shape).contiguous()

        # Start block
        x = self.cnn_block(inputs)

        # Reshape data for classification
        x = x.view(batch_size, -1)

        # Classification
        x = self.classification_block(x)
        return x

    def name(self):
        return "{}".format(AscadCnn.basename())

    @staticmethod
    def save_name(args):
        return "{}".format(AscadCnn.basename())

    @staticmethod
    def basename():
        return AscadCnn.__name__

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.out_shape,
            'input_shape': self.input_shape,
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = AscadCnn(input_shape=checkpoint['input_shape'],
                         out_shape=checkpoint['out_shape'], )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return AscadCnn(out_shape=args['n_classes'],
                        input_shape=args['input_shape'], )
