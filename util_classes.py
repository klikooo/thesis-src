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

MODELS_TABLE = dict(zip([model.basename() for model in MODELS], MODELS))


def get_init_func(basename):
    return MODELS_TABLE[basename].init


def get_save_name(basename, args):
    return MODELS_TABLE[basename].save_name(args)
