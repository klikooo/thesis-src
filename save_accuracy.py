import torch

import util
import numpy as np

from models.load_model import load_model
from util import generate_folder_name
from test import accuracy2
from util_classes import get_save_name, require_domain_knowledge
import os
import json
import sys


def get_ranks(args):
    # Load the data and make it global
    global x_attack, y_attack, dk_plain, key_guesses
    x_attack, y_attack, key_guesses, real_key, dk_plain = load_data(args)

    model_params = {}
    map_accuracy = {}

    folder = "{}/{}/".format(args.models_path, generate_folder_name(args))
    for channel_size in args.channels:
        model_params.update({"channel_size": channel_size})
        for layers in args.layers:
            model_params.update({"num_layers":  layers})
            for kernel in args.kernels:
                model_params.update({"kernel_size": kernel})

                # Calculate the accuracy
                mean_acc = 0.0
                no_data = False
                for run in range(args.runs):
                    model_path = '{}/model_r{}_{}.pt'.format(
                        folder,
                        run,
                        get_save_name(args.network_name, model_params))
                    if not os.path.exists(model_path):
                        print(util.BColors.WARNING + f"Path {model_path} does not exists" + util.BColors.ENDC)
                        no_data = True
                        break
                    print('path={}'.format(model_path))

                    model = load_model(args.network_name, model_path)
                    model.eval()
                    print("Using {}".format(model))
                    model.to(args.device)

                    # Calculate predictions
                    if require_domain_knowledge(args.network_name):
                        _, acc = accuracy2(model, x_attack, y_attack, dk_plain)
                    else:
                        _, acc = accuracy2(model, x_attack, y_attack, None)
                    print('Accuracy: {} - {}%'.format(acc, acc * 100))
                    acc = acc * 100
                    mean_acc = mean_acc + acc
                if not no_data:
                    mean_acc = mean_acc / float(args.runs)
                    map_accuracy.update({f"c_{channel_size}_l{layers}_k{kernel}": mean_acc})
                    print(util.BColors.WARNING + f"Mean accuracy {mean_acc}" + util.BColors.ENDC)

    acc_filename = f"{folder}/acc2_{args.network_name}.json"
    print(acc_filename)
    with open(acc_filename, "w") as acc_file:
        acc_file.write(json.dumps(map_accuracy))


def load_data(args):
    _x_attack, _y_attack, _real_key, _dk_plain, _key_guesses = None, None, None, None, None
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
                                                    'data_set': args.data_set,
                                                    'noise_level': args.noise_level})
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


def run_load(l2_penal):
    args = util.EmptySpace()
    args.use_hw = False
    args.data_set = util.DataSet.RANDOM_DELAY_NORMALIZED
    # args.traces_path = "/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/student-datasets/"
    # args.models_path = "/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/rtubbing/"
    args.traces_path = "/media/rico/Data/TU/thesis/data/"
    args.models_path = "/media/rico/Data/TU/thesis/runs3/"
    args.raw_traces = True
    args.train_size = 40000
    args.validation_size = 1000
    args.attack_size = 9000
    args.use_noise_data = False
    args.epochs = 75
    args.batch_size = 100
    args.lr = 0.0001
    args.l2_penalty = float(l2_penal)
    args.init_weights = "kaiming"
    args.noise_level = 0.0
    args.type_network = 'HW' if args.use_hw else 'ID'
    args.device = torch.device("cuda")
    args.network_name = "VGGNumLayers"
    args.subkey_index = 2
    args.unmask = True
    args.desync = 0
    args.spread_factor = 1
    args.runs = 5

    args.kernels = [20]
    args.layers = [5]
    # args.kernels = [100, 50, 25, 20, 15, 17, 10, 7, 5, 3]
    # args.layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    args.channels = [32]

    get_ranks(args)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_load(0.005)
    else:
        run_load(sys.argv[1])
