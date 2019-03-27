from util_classes import MODELS


def load_model(network_name, model_path):
    for clazz in MODELS:
        if clazz.basename() == network_name:
            return clazz.load_model(model_path)
