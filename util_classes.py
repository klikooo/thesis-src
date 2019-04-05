from models.Alex.KB import KB
from models.Alex.KB2 import KB2
from models.Alex.KB3 import KB3
from models.Alex.KB4 import KB4
from models.Alex.KBS import KBS
from models.Alex.KBS2 import KBS2
from models.Alex.KBS3 import KBS3
from models.Alex.KernelBig import KernelBig
from models.Alex.KernelBigAvg import KernelBigAvg
from models.Alex.KernelBigSmall import KernelBigSmall
from models.Alex.KernelBigSmallAvg import KernelBigSmallAvg
from models.Alex.NumLayers import NumLayers
from models.CNN.NIN import NIN
from models.ConvNet import ConvNet
from models.ConvNetDK import ConvNetDK
from models.ConvNetDPA import ConvNetDPA
from models.ConvNetKernel import ConvNetKernel
from models.ConvNetKernelAscad import ConvNetKernelAscad
from models.ConvNetKernelAscad2 import ConvNetKernelAscad2
from models.ConvNetKernelAvg import ConvNetKernelAvg
from models.ConvNetKernelMasked import ConvNetKernelMasked
from models.ConvNetKernelSmall import ConvNetKernelSmall
from models.ConvNetKernelSmallAvg import ConvNetKernelSmallAvg
from models.CosNet import CosNet
from models.DenseNet import DenseNet
from models.DenseSpreadNet import DenseSpreadNet
from models.SingleConv import SingleConv
from models.SpreadNet import SpreadNet
from models.SpreadV2 import SpreadV2
from models.VGG.KBVGG import KBVGG
from models.VGG.KernelBigSmallVGG import KernelBigSmallVGG
from models.VGG.KernelBigSmallVGGM import KernelBigSmallVGGM
from models.VGG.KernelBigVGG import KernelBigVGG
from models.VGG.KernelBigVGGC import KernelBigVGGC
from models.VGG.KernelBigVGGM import KernelBigVGGM
from models.VGG.MakeSomeNoise import MakeSomeNoise

MODELS = [DenseSpreadNet, DenseNet, SpreadV2,
          SpreadNet, CosNet, ConvNet, ConvNetDK,
          ConvNetDK, ConvNetDPA, ConvNetKernel,
          ConvNetKernelAscad, ConvNetKernelAscad2,
          ConvNetKernelMasked, NIN, KernelBig, NumLayers,
          KernelBigVGG, KernelBigSmallVGG, KernelBigSmall,
          KernelBigAvg, KernelBigSmallAvg, MakeSomeNoise,
          KB, KBS, KB2, KBS2, KBVGG, KBS3, KB3, KB4,
          ConvNetKernelSmall, KernelBigVGGM, KernelBigSmallVGGM,
          SingleConv, ConvNetKernelAvg, ConvNetKernelSmallAvg,
          KernelBigVGGC]
MODELS_TABLE = dict(zip([model.basename() for model in MODELS], MODELS))


def get_init_func(basename):
    return MODELS_TABLE[basename].init


def get_save_name(basename, args):
    return MODELS_TABLE[basename].save_name(args)
