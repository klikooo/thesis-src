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


def load_model(network_name, model_path):
    classes = [DenseSpreadNet, DenseNet, SpreadV2,
               SpreadNet, CosNet, ConvNet, ConvNetDK,
               ConvNetDK, ConvNetDPA, ConvNetKernel,
               ConvNetKernelAscad, ConvNetKernelAscad2,
               ConvNetKernelMasked, NIN]
    for clazz in classes:
        if clazz.basename() == network_name:
            return clazz.load_model(model_path)
    # loader = {
    #     "DenseSpreadNet": DenseSpreadNet.load_model,
    #     "MLPBEST": DenseNet.load_model,
    #     "SpreadV2": SpreadV2.load_spread,
    #     "SpreadNet": SpreadNet.load_spread,
    #     "CosNet": CosNet.load_model,
    #     "ConvNet": ConvNet.load_model,
    #     "ConvNetDK": ConvNetDK.load_model,
    #     "ConvNetDPA": ConvNetDPA.load_model,
    #     "ConvNetKernel": ConvNetKernel.load_model,
    #     "ConvNetKernelAscad": ConvNetKernelAscad.load_model,
    #     "ConvNetKernelAscad2": ConvNetKernelAscad2.load_model,
    #     "ConvNetKernelMasked": ConvNetKernelMasked.load_model,
    #     "NIN": NIN.load_model
    # }
    # return loader[network_name](model_path)
