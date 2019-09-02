from models.Alex.AlexNet import AlexNet
from models.Alex.KB import KB
from models.Alex.KB2 import KB2
from models.Alex.KB3 import KB3
from models.Alex.KB4 import KB4
from models.Alex.KBS import KBS
from models.Alex.KBS2 import KBS2
from models.Alex.KBS3 import KBS3
from models.Alex.KernelBig import KernelBig
from models.Alex.KernelBigAlex import KernelBigAlex
from models.Alex.KernelBigAlexBN import KernelBigAlexBN
from models.Alex.KernelBigAvg import KernelBigAvg
from models.Alex.KernelBigSmall import KernelBigSmall
from models.Alex.KernelBigSmallAvg import KernelBigSmallAvg
from models.Alex.NumLayers import NumLayers
from models.CNN.AscadCnn import AscadCnn
from models.CNN.NIN import NIN
from models.CNN.SmallCNN import SmallCNN
from models.CNN.ZaidCNN import ZaidCNN
from models.CNN.ZaidCNNMasked import ZaidCNNMasked
from models.CNN.ZaidCNNRD import ZaidCNNRD
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
from models.Spread.DenseBatch import DenseBatch
from models.Spread.DenseNet import DenseNet
from models.Spread.DenseNorm import DenseNorm
from models.Spread.DenseSpreadNet import DenseSpreadNet
from models.SingleConv import SingleConv
from models.Spread.SpreadFirstLayer import SpreadFirstLayer
from models.Spread.SpreadNet import SpreadNet
from models.Spread.SpreadV2 import SpreadV2
from models.Spread.SpreadV3 import SpreadV3
from models.VGG.BigChannels import BigChannels
from models.VGG.DK.KernelBigVGGDK import KernelBigVGGDK
from models.VGG.DK.KernelBigVGGMDK import KernelBigVGGMDK
from models.VGG.DK.VGGNumLayers4DK import VGGNumLayers4DK
from models.VGG.KBVGG import KBVGG
from models.VGG.KernelBigSmallVGG import KernelBigSmallVGG
from models.VGG.KernelBigSmallVGGM import KernelBigSmallVGGM
from models.VGG.KernelBigTest import KernelBigTest
from models.VGG.KernelBigTestM import KernelBigTestM
from models.VGG.KernelBigVGG import KernelBigVGG
from models.VGG.KernelBigVGGC import KernelBigVGGC
from models.VGG.KernelBigVGGM import KernelBigVGGM
from models.VGG.KernelBigVGGRDLM import KernelBigVGGRDLM
from models.VGG.KernelBigVGGRDLM2 import KernelBigVGGRDLM2
from models.VGG.MakeSomeNoise import MakeSomeNoise
from models.VGG.NumLayersVGG import NumLayersVGG
from models.VGG.NumLayersVGG2 import NumLayersVGG2
from models.VGG.NumLayersVGG3 import NumLayersVGG3
from models.VGG.VGGMasked import VGGMasked
from models.VGG.VGGNumBlocks import VGGNumBlocks
from models.VGG.VGGNumLayers import VGGNumLayers
from models.VGG.VGGNumLayers2 import VGGNumLayers2
from models.VGG.VGGNumLayers3 import VGGNumLayers3
from models.VGG.VGGNumLayers4 import VGGNumLayers4
from models.makesomenoise.MakeSomeNoise import MakeSomeNoiseReal

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
          KernelBigVGGC, NumLayersVGG, AlexNet, KernelBigVGGRDLM, KernelBigVGGRDLM2,
          NumLayersVGG2, KernelBigVGGDK, KernelBigVGGMDK, NumLayersVGG3,
          KernelBigTest, KernelBigTestM, KernelBigAlex, KernelBigAlexBN,
          VGGNumLayers, VGGNumBlocks, SpreadFirstLayer, BigChannels,
          DenseNorm, DenseBatch, SpreadV3, SmallCNN, MakeSomeNoiseReal,
          VGGNumLayers2, VGGMasked, VGGNumLayers3, VGGNumLayers4, VGGNumLayers4DK,
          AscadCnn, ZaidCNN, ZaidCNNRD, ZaidCNNMasked]
MODELS_TABLE = dict(zip([model.basename() for model in MODELS], MODELS))


def get_init_func(basename):
    return MODELS_TABLE[basename].init


def get_save_name(basename, args):
    return MODELS_TABLE[basename].save_name(args)


###########################
# DOMAIN KNOWLEDGE TABLES #
###########################
MODELS_DK = [ConvNetDK, KernelBigVGGDK, ConvNetDPA, KernelBigVGGMDK, VGGNumLayers4DK]
MODELS_DK_TABLE = dict(zip([model.basename() for model in MODELS_DK], MODELS_DK))


def require_domain_knowledge(basename):
    return basename in MODELS_DK_TABLE
