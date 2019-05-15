import torch.nn


def kaiming_init(m):
    if type(m) == torch.nn.Conv1d:
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        m.bias.data.fill_(0.0)


INIT_WEIGHTS_MAP = {
    "kaiming": kaiming_init
}


def init_weights(model, name):
    print("Init weight with: {}".format(name))
    if name in INIT_WEIGHTS_MAP:
        model.apply(INIT_WEIGHTS_MAP[name])
