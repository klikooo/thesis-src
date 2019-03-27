from util_classes import MODELS


def load_model(network_name, model_path):
    for clazz in MODELS:
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
