import torch.nn


def kaiming_init(m):
    if type(m) == torch.nn.Conv1d:
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        m.bias.data.fill_(0.0)


def xavier_init(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


def kaiming_uniform_init(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv1d:
        torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        m.bias.data.fill_(0.0)


INIT_WEIGHTS_MAP = {
    "kaiming": kaiming_init,
    "xavier": xavier_init,
    "kaiming_uniform": kaiming_uniform_init
}


def init_weights(model, name):
    print("Init weight with: {}".format(name))
    if name in INIT_WEIGHTS_MAP:
        model.apply(INIT_WEIGHTS_MAP[name])
