from multiprocessing import Process

import torch

import util
import numpy as np
import json

from models.load_model import load_model
from util import shuffle_permutation, generate_folder_name, BColors
from test import accuracy, test_with_key_guess_p, create_key_probabilities, test_with_key_probabilities
from util_classes import get_save_name, require_domain_knowledge
import os


def predictions_exist(path, model_name, noise_string):
    file = f'{path}/predictions{noise_string}_model_r0_{model_name}.npy'
    return os.path.exists(file)


def load_predictions(path, model_name, runs, noise_string):
    print(f"{BColors.WARNING}Loading predictions{BColors.ENDC}")
    predictions = []
    filename = f"{path}/predictions{noise_string}_model_r" + "{}_" + f"{model_name}.npy"
    for run in runs:
        predictions.append(np.load(filename.format(run)))
    print("Loaded predictions")
    return predictions


def create_predictions(path, args, network_name, model_params, noise_string):
    print(f"{BColors.WARNING}Creating predictions{BColors.ENDC}")
    x_test, y_test, plain_test, key_test, key_guesses_test = load_data(args)

    # Calculate the predictions before hand
    predictions = []
    sum_acc = 0.0
    for run in args.runs:
        model_path = '{}/model_r{}_{}.pt'.format(
            path,
            run,
            get_save_name(network_name, model_params))
        print('path={}'.format(model_path))

        model = load_model(network_name, model_path)
        model.eval()
        print("Using {}".format(model))
        model.to(args.device)

        # Calculate predictions
        if require_domain_knowledge(network_name):
            prediction, acc = accuracy(model, x_test, y_test, plain_test)
            predictions.append(prediction.cpu().numpy())
        else:
            prediction, acc = accuracy(model, x_test, y_test, None)
            predictions.append(prediction.cpu().numpy())
        sum_acc += acc

        # Save the predictions
        if args.save_predictions:
            predictions_save_file = f'{path}/predictions{noise_string}_' \
                                    f'model_r{run}_{get_save_name(network_name, model_params)}'
            np.save(predictions_save_file, prediction.cpu().numpy())
            print(f"{BColors.WARNING}Saved predictions to {predictions_save_file}.npy{BColors.ENDC}")

    # Save accuracy
    if args.save_predictions:
        mean_acc = sum_acc / len(args.runs)
        print(util.BColors.WARNING + f"Mean accuracy {mean_acc}" + util.BColors.ENDC)
        noise_extension = f'_noise{args.noise_level}' if args.use_noise_data and args.noise_level > 0.0 else ''
        mean_acc_file = f"{path}/acc_{get_save_name(network_name, model_params)}{noise_extension}.acc"
        with open(mean_acc_file, "w") as file:
            file.write(json.dumps(mean_acc))
    return x_test, y_test, plain_test, key_test, key_guesses_test, predictions


# Load data + calc GE
def get_ranks(args, network_name, model_params):
    folder = "{}/{}/".format(args.models_path, generate_folder_name(args))
    model_name = get_save_name(network_name, model_params)

    # Load the data and make it global
    global x_attack, y_attack, dk_plain, key_guesses

    # Create file strings
    noise_string = f'_noise{args.noise_level}' if args.use_noise_data and args.noise_level > 0.0 else ''

    # Load predictions
    if args.load_predictions and predictions_exist(folder, model_name, noise_string):
        predictions = load_predictions(folder, model_name, args.runs, noise_string)

        # Skip loading the traces
        args.load_traces = False
        _, _, dk_plain, real_key, key_guesses = load_data(args)
    # Create predictions
    else:
        args.load_traces = True
        x_attack, y_attack, dk_plain, real_key, key_guesses, predictions = create_predictions(folder,
                                                                                              args,
                                                                                              network_name,
                                                                                              model_params,
                                                                                              noise_string)

    # Check if it is only one run, if so don't do multi threading
    if len(args.runs) == 1:
        threaded_run_test(args, predictions[0], folder, args.runs[0], network_name, model_params, real_key)
    else:
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


def load_data(args):
    return util.load_test_data(args)


def threaded_run_test(args, prediction, folder, run, network_name, model_params, real_key):
    print("Creating key probabilities")
    key_probabilities = create_key_probabilities(key_guesses, prediction, args.attack_size, args.use_hw)

    y = []
    for exp_i in range(args.num_exps):
        # Select permutation
        permutation = args.permutations[exp_i]
        key_probabilities_shuffled = shuffle_permutation(permutation, key_probabilities)

        # Test the data
        x_exp, y_exp, k_guess = test_with_key_probabilities(key_probabilities_shuffled, real_key)
        print(f'{exp_i}: Key rank: {y_exp[-1]}, guess: {k_guess}')
        y.append(y_exp)

    # Shuffle the data using same permutation  for n_exp and calculate mean for GE of the model
    # y = []
    # for exp_i in range(args.num_exps):
    #     # Select permutation
    #     permutation = args.permutations[exp_i]
    #
    #     # Shuffle data
    #     predictions_shuffled = shuffle_permutation(permutation, np.array(prediction))
    #     key_guesses_shuffled = shuffle_permutation(permutation, key_guesses)
    #
    #     # Test the data
    #     x_exp, y_exp, k_guess = test_with_key_guess_p(key_guesses_shuffled, predictions_shuffled,
    #                                                   attack_size=args.attack_size,
    #                                                   real_key=real_key,
    #                                                   use_hw=args.use_hw)
    #     print(f'{exp_i}: Key rank: {y_exp[-1]}, guess: {k_guess}')
    #     y.append(y_exp)

    # Calculate the mean over the experiments
    y = np.mean(y, axis=0)
    if args.use_noise_data:
        save_path = '{}model_r{}_{}_noise{}.exp'.format(folder, run,
                                                        get_save_name(network_name, model_params),
                                                        args.noise_level)
    else:
        save_path = '{}model_r{}_{}.exp'.format(folder, run, get_save_name(network_name, model_params))
    print("Save path {}".format(save_path))
    util.save_np(save_path, y, f="%f")


def run_load(args):
    args.type_network = 'HW' if args.use_hw else 'ID'

    args.device = torch.device("cuda") if args.load_predictions else torch.device("cpu")

    args.permutations = util.generate_permutations(args.num_exps, args.attack_size)

    model_params = {}

    # Test the networks that were specified
    for net_name in args.network_names:
        def kernel_lambda(x): model_params.update({"kernel_size": x})

        def channel_lambda(x): model_params.update({"channel_size": x})

        def layers_lambda(x): model_params.update({"num_layers": x})

        model_params.update({"max_pool": args.max_pool})

        util.loop_at_least_once(args.kernel_sizes, kernel_lambda, lambda: (
            util.loop_at_least_once(args.channel_sizes, channel_lambda, lambda: (
                util.loop_at_least_once(args.num_layers, layers_lambda, lambda: (
                    print(model_params),
                    get_ranks(args, net_name, model_params))
                                        ))
                                    )
        ))
