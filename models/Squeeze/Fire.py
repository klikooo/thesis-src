import torch
import torch.nn as nn
import torch.nn.init as init
from util import device


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv1d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv1d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv1d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', input_shape=700, output_shape=1000):
        super(SqueezeNet, self).__init__()
        self.output_shape = output_shape
        self.input_shape = input_shape
        if version == '1_0':
            self.features = nn.Sequential(
                # nn.Conv1d(1, 1, kernel_size=1, stride=2, padding=0),
                # nn.Linear(350, output_shape),

                nn.Conv1d(1, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            ).to(device)
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool1d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv1d(512, self.output_shape, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        ).to(device)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.to(device).view(batch_size, 1, self.input_shape).contiguous()
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def name(args):
        return "{}".format(SqueezeNet.basename())

    @staticmethod
    def save_name(args):
        return "{}".format(SqueezeNet.basename())

    @staticmethod
    def basename():
        return SqueezeNet.__name__

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'out_shape': self.output_shape,
            'input_shape': self.output_shape,
        }, path)

    @staticmethod
    def load_model(file):
        checkpoint = torch.load(file)

        model = SqueezeNet(output_shape=checkpoint['out_shape'],
                           input_shape=checkpoint['input_shape'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @staticmethod
    def init(args):
        return SqueezeNet(output_shape=args['n_classes'],
                          input_shape=args['input_shape'])
