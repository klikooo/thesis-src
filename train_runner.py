from decimal import Decimal

from models.Spread.SpreadNet import SpreadNet

from util import save_model, load_data_set, DataSet, save_loss_acc
from train import train, train_dk

import numpy as np


def run(use_hw, runs, train_size, epochs, batch_size, lr, subkey_index, spread_factor, init, input_shape, checkpoints,
        traces_path,
        model_save_path,
        data_set,
        raw_traces,
        desync,
        validation_size,
        loss_function,
        channel_size,
        num_layers,
        l2_penalty,
        unmask=False,
        domain_knowledge=False,
        kernel_size=None,
        use_noise_data=False):
    sub_key_index = subkey_index

    # Select the number of classes to use depending on hw
    n_classes = 9 if use_hw else 256

    # Save the models to this folder
    dir_name = '{}/subkey_{}/{}{}{}_SF{}_E{}_BZ{}_LR{}{}/train{}'.format(
        str(data_set),
        sub_key_index,
        '' if unmask or data_set is not DataSet.ASCAD else 'masked/',
        '' if desync is 0 else 'desync{}/'.format(desync),
        'HW' if use_hw else 'ID',
        spread_factor,
        epochs,
        batch_size,
        '%.2E' % Decimal(lr),
        '' if np.math.ceil(l2_penalty) <= 0 else '_L2_{}'.format(l2_penalty),
        train_size,
    )

    # Arguments for loading data
    load_args = {"unmask": unmask,
                 "use_hw": use_hw,
                 "traces_path": traces_path,
                 "sub_key_index": sub_key_index,
                 "raw_traces": raw_traces,
                 "size": train_size + validation_size,
                 "domain_knowledge": True,
                 "desync": desync,
                 "use_noise_data": use_noise_data}

    # Load data and chop into the desired sizes
    load_function = load_data_set(data_set)
    print(load_args)
    x_train, y_train, plain = load_function(load_args)
    x_validation = x_train[train_size:train_size + validation_size]
    y_validation = y_train[train_size:train_size + validation_size]
    x_train = x_train[0:train_size]
    y_train = y_train[0:train_size]

    print('Shape x: {}'.format(np.shape(x_train)))

    # Arguments for initializing the model
    init_args = {"sf": spread_factor,
                 "input_shape": input_shape,
                 "n_classes": n_classes,
                 "kernel_size": kernel_size,
                 "channel_size": channel_size,
                 "num_layers": num_layers
                 }

    # Do the runs
    for i in range(runs):
        # Initialize the network and train it
        network = init(init_args)

        # Where the file is stored
        filename = 'model_r{}_{}'.format(i, network.name())
        model_save_file = '{}/{}/{}.pt'.format(model_save_path, dir_name, filename)

        print('Training with learning rate: {}, desync {}'.format(lr, desync))

        if domain_knowledge:
            # TODO: save losses and accuracies for this
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
            network, res = train(x_train, y_train,
                                 train_size=train_size,
                                 x_validation=x_validation,
                                 y_validation=y_validation,
                                 validation_size=validation_size,
                                 network=network,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 lr=lr,
                                 checkpoints=checkpoints,
                                 save_path=model_save_file,
                                 loss_function=loss_function,
                                 l2_penalty=l2_penalty
                                 )
            save_loss_acc(model_save_file, filename, res)

        # Make sure don't mess with our min/max of the spread network
        if isinstance(network, SpreadNet):
            network.training = False

        # Save the final model
        save_model(network, model_save_file)
