import torch


def cyclicLR(optimizer, args):
    return torch.optim.lr_scheduler.CyclicLR(optimizer,
                                             cycle_momentum=False,
                                             base_lr=args['base_lr'],
                                             max_lr=args['max_lr'])


map_schedulers = {
    "CyclicLR": cyclicLR
}


def get_scheduluer(name):
    return map_schedulers[name]
