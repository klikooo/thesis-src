from decimal import Decimal

from models.DenseNet import DenseNet
from models.SpreadNet import SpreadNet
from models.DenseSpreadNet import DenseSpreadNet

import os
import torch

from ascad import load_ascad
from train import train
from util import SBOX


def run(use_hw, runs, train_size, epochs, batch_size, lr, subkey_index, spread_factor, init, input_shape, checkpoints,
        unmask=False):
    path = '/media/rico/Data/TU/thesis'

    sub_key_index = subkey_index

    # Select the number of classes to use depending on hw
    n_classes = 9 if use_hw else 256
    traces_file = '{}/data/ASCAD_{}.h5'.format(path, sub_key_index)

    # Save the ranks to a file
    dir_name = 'subkey_{}/{}_SF{}_E{}_BZ{}_LR{}/train{}'.format(
        sub_key_index,
        'HW' if use_hw else 'ID',
        spread_factor,
        epochs,
        batch_size,
        '%.2E' % Decimal(lr),
        train_size
    )

    # Load data
    (x_profiling, y_profiling), (_, _), (metadata_profiling, _) = load_ascad(traces_file, load_metadata=True)
    if unmask:
        y_profiling = [y_profiling[i] ^ metadata_profiling[i]['masks'][sub_key_index-2] for i in range(len(y_profiling))]

    # i = 100
    # j = sub_key_index
    # k = metadata_profiling[i]['key'][j]
    # y = y_profiling[i]
    # p = metadata_profiling[i]['plaintext'][j]
    # s = SBOX[p ^ k]
    # m = metadata_profiling[i]['masks'][j]
    # r = [s ^ mask for mask in metadata_profiling[i]['masks']]
    # print('Y label: {}'.format(y))
    # print('Key: {}'.format(k))
    # print('Plain: {}'.format(p))
    # print('Sbox out: {} '.format(s))
    # print('mask : {}'.format(m))
    # print('Res:  {}'.format(s^m))
    # print('Res:  {}'.format(r))

    # print(metadata_profiling[:]['masks'][15])
    # exit(1)




    # Do the runs
    for i in range(runs):
        # Choose which network to use
        # network = SpreadNet(spread_factor=spread_factor, input_shape=700, out_shape=n_classes)
        network = init(spread_factor, input_shape, n_classes)
        # network = TestNet(pr_shape=460, sbox_shape=500, n_classes=n_classes)
        # network = DenseNet(input_shape=700, n_classes=n_classes)

        network = train(x_profiling, y_profiling,
                        train_size=train_size,
                        network=network,
                        epochs=epochs,
                        batch_size=batch_size,
                        use_hw=use_hw,
                        lr=lr
                        )
        # Make sure don't mess with our min/max of the spread network
        if isinstance(network, SpreadNet):
            network.training = False

        type_network = network.name()

        model_save_file = '{}/runs/{}/model_r{}_{}.pt'.format(path, dir_name, i, type_network)
        os.makedirs(os.path.dirname(model_save_file), exist_ok=True)

        if isinstance(network, SpreadNet):
            network.save(model_save_file)
        elif isinstance(network, DenseSpreadNet):
            network.save(model_save_file)
        elif isinstance(network, DenseNet):
            network.save(model_save_file)
        else:
            torch.save(network.state_dict(), model_save_file)
