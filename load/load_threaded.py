from decimal import Decimal
from multiprocessing import Process

import torch

import util
import numpy as np

from models.load_model import load_model
from util import load_ascad, shuffle_permutation, req_dk, hot_encode
from test import accuracy, test_with_key_guess_p


def get_ranks(args, network_name, kernel_size_string=""):
    # Load the data and make it global
    global x_attack, y_attack, dk_plain, key_guesses
    x_attack, y_attack, key_guesses, real_key, dk_plain = load_data(args, network_name)

    folder = '{}/{}/subkey_{}/{}{}{}_SF{}_' \
             'E{}_BZ{}_LR{}/train{}/'.format(
                args.models_path,
                str(args.data_set),
                args.subkey_index,
                '' if args.unmask else 'masked/',
                '' if args.desync is 0 else 'desync{}/'.format(args.desync),
                args.type_network,
                args.spread_factor,
                args.epochs,
                args.batch_size,
                '%.2E' % Decimal(args.lr),
                args.train_size)

    # Calculate the predictions before hand
    predictions = []
    for run in args.runs:
        model_path = '{}/model_r{}_{}{}.pt'.format(
            folder,
            run,
            network_name,
            kernel_size_string)
        print('path={}'.format(model_path))

        model = load_model(network_name, model_path)
        model.eval()
        print("Using {}".format(model))
        model.to(args.device)

        # Calculate predictions
        prediction = accuracy(model, x_attack, y_attack, dk_plain)
        predictions.append(prediction.cpu().numpy())

    # Start a thread for each run
    processes = []
    for i, run in enumerate(args.runs):
        p = Process(target=threaded_run_test, args=(args, predictions[i], folder, run,
                                                    network_name, kernel_size_string, real_key))
        processes.append(p)
        p.start()
    # Wait for them to finish
    for p in processes:
        p.join()
        print('Joined process')


def load_data(args, network_name):
    _x_attack, _y_attack, _real_key, _dk_plain, _key_guesses = None, None, None, None, None
    if args.data_set == util.DataSet.ASCAD:
        args.trace_file = '{}/ASCAD/ASCAD_{}_desync{}.h5'.format(args.traces_path, args.subkey_index, args.desync)
        (_, _), (_x_attack, _y_attack), (_metadata_profiling, _metadata_attack) = load_ascad(
            args.trace_file, load_metadata=True)
        _key_guesses = util.load_csv('{}/ASCAD/key_guesses.csv'.format(args.traces_path),
                                     delimiter=' ',
                                     dtype=np.int,
                                     start=0,
                                     size=args.attack_size)
        # Manipulate the data
        _x_attack = _x_attack[:args.attack_size]
        _y_attack = _y_attack[:args.attack_size]
        if args.unmask:
            if args.use_hw:
                _y_attack = np.array([_y_attack[i] ^ _metadata_attack[i]['masks'][0] for i in range(len(_y_attack))])
            else:
                _y_attack = np.array(
                    [util.HW[_y_attack[i] ^ _metadata_attack[i]['masks'][0]] for i in range(len(_y_attack))])
        _real_key = _metadata_attack[0]['key'][args.subkey_index]

        # Load additional plaintexts
        if network_name in req_dk:
            _dk_plain = _metadata_attack[:]['plaintext'][:, args.subkey_index]
            _dk_plain = hot_encode(_dk_plain, 9 if args.use_hw else 256, dtype=np.float)
    else:
        loader = util.load_data_set(args.data_set)
        total_x_attack, total_y_attack, plain = loader({'use_hw': args.use_hw,
                                                        'traces_path': args.traces_path,
                                                        'raw_traces': args.raw_traces,
                                                        'start': args.train_size + args.validation_size,
                                                        'size': args.attack_size,
                                                        'domain_knowledge': True,
                                                        'use_noise_data': args.use_noise_data})
        print('Loading key guesses')
        data_set_name = str(args.data_set)
        _key_guesses = util.load_csv('{}/{}/Value/key_guesses_ALL_transposed.csv'.format(
            args.traces_path,
            data_set_name),
            delimiter=' ',
            dtype=np.int,
            start=args.train_size + args.validation_size,
            size=args.attack_size)
        _real_key = util.load_csv('{}/{}/secret_key.csv'.format(args.traces_path, data_set_name),
                                  dtype=np.int)

        _x_attack = total_x_attack
        _y_attack = total_y_attack

    return _x_attack, _y_attack, _key_guesses, _real_key, _dk_plain


def threaded_run_test(args, prediction, folder, run, network_name, kernel_size_string, real_key):
    # Shuffle the data using same permutation  for n_exp and calculate mean for GE of the model
    y = []
    for exp_i in range(args.num_exps):
        # Select permutation
        permutation = args.permutations[exp_i]

        # Shuffle data
        predictions_shuffled = shuffle_permutation(permutation, np.array(prediction))
        key_guesses_shuffled = shuffle_permutation(permutation, key_guesses)

        # Test the data
        x_exp, y_exp = test_with_key_guess_p(key_guesses_shuffled, predictions_shuffled,
                                             attack_size=args.attack_size,
                                             real_key=real_key,
                                             use_hw=args.use_hw)
        y.append(y_exp)

    # Calculate the mean over the experiments
    y = np.mean(y, axis=0)
    save_path = '{}model_r{}_{}{}.exp'.format(folder, run, network_name, kernel_size_string)
    print("Save path {}".format(save_path))
    util.save_np(save_path, y, f="%f")


def run_load(args):
    args.type_network = 'HW' if args.use_hw else 'ID'

    args.device = torch.device("cuda")

    args.permutations = util.generate_permutations(args.num_exps, args.attack_size)

    # Test the networks that were specified
    for net_name in args.network_names:
        if net_name in util.req_kernel_size:
            for kernel_size in args.kernel_sizes:
                kernel_string = "_k{}".format(kernel_size)
                get_ranks(args, net_name, kernel_string)
        else:
            get_ranks(args, net_name)
