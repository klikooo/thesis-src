from models.CosNet import CosNet
from models.DenseNet import DenseNet
from models.DenseSpreadNet import DenseSpreadNet
from models.SpreadNet import SpreadNet
from models.SpreadV2 import SpreadV2


def load_model(network_name, model_path):
    if "DenseSpreadNet" in network_name:
        model = DenseSpreadNet.load_model(model_path)
    elif "MLP" in network_name:
        model = DenseNet.load_model(model_path)
    elif "SpreadV2" in network_name:
        model = SpreadV2.load_spread(model_path)
    # elif "SpreadNet" in network_name:
    #     model = SpreadNetIn.load_spread(model_path)
    elif "SpreadNet" in network_name:
        model = SpreadNet.load_spread(model_path)
    elif "CosNet" in network_name:
        model = CosNet.load_model(model_path)
    else:
        raise Exception("Unknown model")
    return model
