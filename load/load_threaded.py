from multiprocessing import Process

import torch

import util
import numpy as np

from models.load_model import load_model
from util import load_ascad, shuffle_permutation, hot_encode, generate_folder_name
from test import accuracy, test_with_key_guess_p
from util_classes import get_save_name, require_domain_knowledge


def get_ranks(args, network_name, model_params):
    # Load the data and make it global
    global x_attack, y_attack, dk_plain, key_guesses
    x_attack, y_attack, key_guesses, real_key, dk_plain = load_data(args, network_name)

    folder = "{}/{}/".format(args.models_path, generate_folder_name(args))

    # Calculate the predictions before hand
    predictions = []
    for run in args.runs:
        model_path = '{}/model_r{}_{}.pt'.format(
            folder,
            run,
            get_save_name(network_name, model_params))
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
                                                    network_name, model_params, real_key))
        processes.append(p)
        p.start()
    # Wait for them to finish
    for p in processes:
        p.join()
        print('Joined process')


def load_data(args, network_name):
    _x_attack, _y_attack, _real_key, _dk_plain, _key_guesses = None, None, None, None, None
    argz = {'use_hw': args.use_hw,
            'traces_path': args.traces_path,
            'raw_traces': args.raw_traces,
            'start': args.train_size + args.validation_size,
            'size': args.attack_size,
            'train_size': args.train_size,
            'validation_size': args.validation_size,
            'domain_knowledge': True,
            'use_noise_data': args.use_noise_data,
            'data_set': args.data_set,
            'sub_key_index': args.subkey_index,
            'desync': args.desync,
            'unmask': args.unmask}

    if args.data_set == util.DataSet.ASCAD:
        _x_attack, _y_attack, _plain, _real_key, _key_guesses = util.load_ascad_test_traces(argz)
    elif args.data_set == util.DataSet.ASCAD_NORMALIZED:
        _x_attack, _y_attack, _key_guesses, _real_key = util.load_ascad_normalized_test_traces(argz)
    elif args.data_set == util.DataSet.SIM_MASK:
        _x_attack, _y_attack, _key_guesses, _real_key = util.load_sim_mask_test_traces(argz)
    elif args.data_set == util.DataSet.RANDOM_DELAY_LARGE:
        ###################
        # Load the traces #
        ###################
        loader = util.load_data_set(args.data_set)
        total_x_attack, total_y_attack, plain = loader({'use_hw': args.use_hw,
                                                        'traces_path': args.traces_path,
                                                        'raw_traces': args.raw_traces,
                                                        'start': args.train_size + args.validation_size,
                                                        'size': args.attack_size,
                                                        'domain_knowledge': True,
                                                        'use_noise_data': args.use_noise_data,
                                                        'data_set': args.data_set})
        print('Loading key guesses')

        ####################################
        # Load the key guesses and the key #
        ####################################
        data_set_name = str(args.data_set)
        _key_guesses = util.load_random_delay_large_key_guesses(args.traces_path,
                                                                args.train_size + args.validation_size,
                                                                args.attack_size)
        _real_key = util.load_csv('{}/{}/secret_key.csv'.format(args.traces_path, data_set_name),
                                  dtype=np.int)

        _x_attack = total_x_attack
        _y_attack = total_y_attack

    else:
        ###################
        # Load the traces #
        ###################
        loader = util.load_data_set(args.data_set)
        total_x_attack, total_y_attack, plain = loader({'use_hw': args.use_hw,
                                                        'traces_path': args.traces_path,
                                                        'raw_traces': args.raw_traces,
                                                        'start': args.train_size + args.validation_size,
                                                        'size': args.attack_size,
                                                        'domain_knowledge': True,
                                                        'use_noise_data': args.use_noise_data,
                                                        'data_set': args.data_set})
        if plain is not None:
            _dk_plain = torch.from_numpy(plain).cuda()
        print('Loading key guesses')

        ####################################
        # Load the key guesses and the key #
        ####################################
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


def threaded_run_test(args, prediction, folder, run, network_name, model_params, real_key):
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
    save_path = '{}model_r{}_{}.exp'.format(folder, run, get_save_name(network_name, model_params))
    print("Save path {}".format(save_path))
    util.save_np(save_path, y, f="%f")


def run_load(args):
    args.type_network = 'HW' if args.use_hw else 'ID'

    args.device = torch.device("cuda")

    args.permutations = util.generate_permutations(args.num_exps, args.attack_size)

    model_params = {}

    # Test the networks that were specified
    for net_name in args.network_names:
        def kernel_lambda(x): model_params.update({"kernel_size": x})

        def channel_lambda(x): model_params.update({"channel_size": x})

        def layers_lambda(x): model_params.update({"num_layers": x})

        util.loop_at_least_once(args.kernel_sizes, kernel_lambda, lambda: (
            util.loop_at_least_once(args.channel_sizes, channel_lambda, lambda: (
                util.loop_at_least_once(args.num_layers, layers_lambda, lambda: (
                    print(model_params),
                    get_ranks(args, net_name, model_params))
                                        ))
                                    )
        ))
