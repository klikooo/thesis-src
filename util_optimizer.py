import torch


def adam(params, args):
    return torch.optim.Adam(params, lr=args['lr'], weight_decay=args['l2'])


def rms_prop(params, args):
    return torch.optim.RMSprop(params, lr=args['lr'], weight_decay=args['l2'])


map_optimizers = {
    "Adam": adam,
    "RMSprop": rms_prop
}


def get_optimizer(name):
    return map_optimizers[name]
