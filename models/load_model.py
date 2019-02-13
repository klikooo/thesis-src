from models.ConvNet import ConvNet
from models.ConvNetDK import ConvNetDK
from models.CosNet import CosNet
from models.DenseNet import DenseNet
from models.DenseSpreadNet import DenseSpreadNet
from models.SpreadNet import SpreadNet
from models.SpreadV2 import SpreadV2


def load_model(network_name, model_path):
    loader = {
        "DenseSpreadNet": DenseSpreadNet.load_model,
        "MLP": DenseNet.load_model,
        "SpreadV2": SpreadV2.load_spread,
        "SpreadNet": SpreadNet.load_spread,
        "CosNet": CosNet.load_model,
        "ConvNet": ConvNet.load_model,
        "ConvNetDK": ConvNetDK.load_model,
    }
    # if "DenseSpreadNet" in network_name:
    #     model = DenseSpreadNet.load_model(model_path)
    # elif "MLP" in network_name:
    #     model = DenseNet.load_model(model_path)
    # elif "SpreadV2" in network_name:
    #     model = SpreadV2.load_spread(model_path)
    # # elif "SpreadNet" in network_name:
    # #     model = SpreadNetIn.load_spread(model_path)
    # elif "SpreadNet" in network_name:
    #     model = SpreadNet.load_spread(model_path)
    # elif "CosNet" in network_name:
    #     model = CosNet.load_model(model_path)
    # elif
    # elif "ConvNet" in network_name:
    #     model = ConvNet.load_model(model_path)
    # else:
    #     raise Exception("Unknown model")
    return loader[network_name](model_path)
