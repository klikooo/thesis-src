from models.Spread.SpreadNet import SpreadNet

from util import save_model, load_data_set, save_loss_acc, generate_folder_name, load_test_data, BColors
from train import train, train_dk2
from test import accuracy

import numpy as np
import json

from util_init import init_weights
from sklearn.preprocessing import StandardScaler


def run(args):
    # Save the models to this folder
    dir_name = generate_folder_name(args)

    # Arguments for loading data
    load_args = {"unmask": args.unmask,
                 "use_hw": args.use_hw,
                 "traces_path": args.traces_path,
                 "sub_key_index": args.subkey_index,
                 "raw_traces": args.raw_traces,
                 "size": args.train_size + args.validation_size,
                 "train_size": args.train_size,
                 "validation_size": args.validation_size,
                 "domain_knowledge": True,
                 "desync": args.desync,
                 "use_noise_data": args.use_noise_data,
                 "start": 0,
                 "data_set": args.data_set}

    # Load data and chop into the desired sizes
    load_function = load_data_set(args.data_set)
    print(load_args)
    x_train, y_train, plain = load_function(load_args)
    x_validation = x_train[args.train_size:args.train_size + args.validation_size]
    y_validation = y_train[args.train_size:args.train_size + args.validation_size]
    x_train = x_train[0:args.train_size]
    y_train = y_train[0:args.train_size]
    p_train = None
    p_validation = None
    if plain is not None:
        p_train = plain[0:args.train_size]
        p_validation = plain[args.train_size:args.train_size + args.validation_size]

    print('Shape x: {}'.format(np.shape(x_train)))

    # Arguments for initializing the model
    init_args = {"sf": args.spread_factor,
                 "input_shape": args.input_shape,
                 "n_classes": 9 if args.use_hw else 256,
                 "kernel_size": args.kernel_size,
                 "channel_size": args.channel_size,
                 "num_layers": args.num_layers,
                 "max_pool": args.max_pool
                 }

    # Load data for creating + saving predictions
    x_test, y_test, plain_test, key_test, _key_guesses_test = None, None, None, None, None
    sum_acc = 0.0
    if args.create_predictions:
        args.noise_level = 0.0
        args.load_traces = True
        x_test, y_test, plain_test, key_test, _key_guesses_test = load_test_data(args)
        if not args.domain_knowledge:
            plain_test = None

    # Normalize data if asked for
    if args.normalize:
        print("Normalizing traces...")
        scale = StandardScaler()
        x_train = scale.fit_transform(x_train)
        x_validation = scale.transform(x_validation)
        if args.create_predictions:
            x_test = scale.transform(x_test)
        print("Done normalizing")

    # Convert scheduler args
    if args.scheduler is not None and type(args.scheduler_args) is str:
        args.scheduler_args = json.loads(args.scheduler_args)

    # Folder path
    path = f"{args.model_save_path}/{dir_name}/"
    model_name = ""

    # Do the runs
    for i in range(args.runs):
        # Initialize the network and the weights
        network = args.init(init_args)
        init_weights(network, args.init_weights)

        # Filename of the model + the folder
        filename = 'model_r{}_{}'.format(i, network.name())
        model_save_file = '{}/{}.pt'.format(path, filename)
        model_name = network.name()

        print('Training with learning rate: {}, desync {}'.format(args.lr, args.desync))

        if args.domain_knowledge:
            network, res = train_dk2(x_train, y_train, p_train,
                                     train_size=args.train_size,
                                     x_validation=x_validation,
                                     y_validation=y_validation,
                                     p_validation=p_validation,
                                     validation_size=args.validation_size,
                                     network=network,
                                     epochs=args.epochs,
                                     batch_size=args.batch_size,
                                     lr=args.lr,
                                     checkpoints=args.checkpoints,
                                     save_path=model_save_file,
                                     loss_function=args.loss_function,
                                     l2_penalty=args.l2_penalty,
                                     )
        else:
            network, res = train(x_train, y_train,
                                 train_size=args.train_size,
                                 x_validation=x_validation,
                                 y_validation=y_validation,
                                 validation_size=args.validation_size,
                                 network=network,
                                 epochs=args.epochs,
                                 batch_size=args.batch_size,
                                 lr=args.lr,
                                 checkpoints=args.checkpoints,
                                 save_path=model_save_file,
                                 loss_function=args.loss_function,
                                 l2_penalty=args.l2_penalty,
                                 optimizer=args.optimizer,
                                 scheduler=args.scheduler,
                                 scheduler_args=args.scheduler_args,
                                 )
        # Save the results of the accuracy and loss during training
        save_loss_acc(model_save_file, filename, res)

        # Make sure don't mess with our min/max of the spread network
        if isinstance(network, SpreadNet):
            network.training = False

        # Save the final model
        save_model(network, model_save_file)
        print(f"Saved model to {model_save_file}")

        # Create + save predictions
        if args.create_predictions:
            predictions, acc = accuracy(network, x_test, y_test, plain_test)
            predictions_save_file = f'{path}/predictions_{filename}'
            np.save(predictions_save_file, predictions.cpu().numpy())
            print(f"{BColors.WARNING}Saved predictions to {predictions_save_file}.npy{BColors.ENDC}")
            sum_acc += acc

    # Save accuracy
    if args.create_predictions:
        mean_acc = sum_acc / len(range(args.runs))
        print(BColors.WARNING + f"Mean accuracy {mean_acc}" + BColors.ENDC)
        noise_extension = f'_noise{args.noise_level}' if args.use_noise_data and args.noise_level > 0.0 else ''
        mean_acc_file = f"{path}/acc_{model_name}{noise_extension}.acc"
        with open(mean_acc_file, "w") as file:
            file.write(json.dumps(mean_acc))
