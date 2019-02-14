from decimal import Decimal

from models.SpreadNet import SpreadNet

from util import save_model, load_data_set
from train import train, train_dk

import numpy as np


def run(use_hw, runs, train_size, epochs, batch_size, lr, subkey_index, spread_factor, init, input_shape, checkpoints,
        traces_path,
        model_save_path,
        data_set,
        raw_traces,
        unmask=False,
        domain_knowledge=False):
    sub_key_index = subkey_index

    # Select the number of classes to use depending on hw
    n_classes = 9 if use_hw else 256

    # Save the models to this folder
    dir_name = '{}/subkey_{}/{}{}_SF{}_E{}_BZ{}_LR{}/train{}'.format(
        str(data_set),
        sub_key_index,
        '' if unmask else 'masked/',
        'HW' if use_hw else 'ID',
        spread_factor,
        epochs,
        batch_size,
        '%.2E' % Decimal(lr),
        train_size
    )

    # Arguments for loading data
    load_args = {"unmask": unmask,
                 "use_hw": use_hw,
                 "traces_path": traces_path,
                 "sub_key_index": sub_key_index,
                 "raw_traces": raw_traces,
                 "size": train_size,
                 "domain_knowledge": True}

    # Load data
    load_function = load_data_set(data_set)
    print(load_args)
    x_train, y_train, plain = load_function(load_args)

    print('Shape x: {}'.format(np.shape(x_train)))

    # Arguments for initializing the model
    init_args = {"sf": spread_factor,
                 "input_shape": input_shape,
                 "n_classes": n_classes
                 }

    # Do the runs
    for i in range(runs):
        # Initialize the network and train it
        network = init(init_args)

        # Where the file is stored
        model_save_file = '{}/{}/model_r{}_{}.pt'.format(model_save_path, dir_name, i, network.name())

        if domain_knowledge:
            network = train_dk(x_train, y_train,
                               train_size=train_size,
                               network=network,
                               epochs=epochs,
                               batch_size=batch_size,
                               lr=lr,
                               checkpoints=checkpoints,
                               save_path=model_save_file,
                               plain=plain
                               )
        else:
            network = train(x_train, y_train,
                            train_size=train_size,
                            network=network,
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=lr,
                            checkpoints=checkpoints,
                            save_path=model_save_file
                            )

        # Make sure don't mess with our min/max of the spread network
        if isinstance(network, SpreadNet):
            network.training = False

        # Save the final model
        save_model(network, model_save_file)
