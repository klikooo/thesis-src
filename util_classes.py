from models.CNN.NIN import NIN
from models.ConvNet import ConvNet
from models.ConvNetDK import ConvNetDK
from models.ConvNetDPA import ConvNetDPA
from models.ConvNetKernel import ConvNetKernel
from models.ConvNetKernelAscad import ConvNetKernelAscad
from models.ConvNetKernelAscad2 import ConvNetKernelAscad2
from models.ConvNetKernelMasked import ConvNetKernelMasked
from models.CosNet import CosNet
from models.DenseNet import DenseNet
from models.DenseSpreadNet import DenseSpreadNet
from models.SpreadNet import SpreadNet
from models.SpreadV2 import SpreadV2

MODELS = [DenseSpreadNet, DenseNet, SpreadV2,
          SpreadNet, CosNet, ConvNet, ConvNetDK,
          ConvNetDK, ConvNetDPA, ConvNetKernel,
          ConvNetKernelAscad, ConvNetKernelAscad2,
          ConvNetKernelMasked, NIN]


def get_init_func(basename):
    for model in MODELS:
        if model.basename() == basename:
            return model.init
