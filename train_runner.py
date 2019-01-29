from decimal import Decimal

from models.CosNet import CosNet
from models.DenseNet import DenseNet
from models.SpreadNet import SpreadNet
from models.DenseSpreadNet import DenseSpreadNet

import os
import torch

from util import load_ascad
from train import train

import numpy as np


def run(use_hw, runs, train_size, epochs, batch_size, lr, subkey_index, spread_factor, init, input_shape, checkpoints,
        traces_path,
        model_save_path,
        unmask=False):

    sub_key_index = subkey_index

    # Select the number of classes to use depending on hw
    n_classes = 9 if use_hw else 256
    traces_file = '{}/ASCAD_{}.h5'.format(traces_path, sub_key_index)

    # Save the models to this folder
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
        y_profiling = np.array(
            [y_profiling[i] ^ metadata_profiling[i]['masks'][sub_key_index-2] for i in range(len(y_profiling))])

    init_args = {"sf": spread_factor,
                 "input_shape": input_shape,
                 "n_classes": n_classes
                 }

    # Do the runs
    for i in range(runs):
        # Initialize the network and train it
        network = init(init_args)
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

        # Make sure the directory where the model should be saved exists
        model_save_file = '{}/{}/model_r{}_{}.pt'.format(model_save_path, dir_name, i, type_network)
        os.makedirs(os.path.dirname(model_save_file), exist_ok=True)

        # Save the model
        if isinstance(network, SpreadNet):
            network.save(model_save_file)
        elif isinstance(network, DenseSpreadNet):
            network.save(model_save_file)
        elif isinstance(network, DenseNet):
            network.save(model_save_file)
        elif isinstance(network, CosNet):
            network.save(model_save_file)
        else:
            torch.save(network.state_dict(), model_save_file)
