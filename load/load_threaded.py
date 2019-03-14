from decimal import Decimal
from multiprocessing import Process

import torch

import util
import numpy as np

from models.load_model import load_model
from util import load_ascad, shuffle_permutation, req_dk, hot_encode
from test import accuracy, test_with_key_guess_p


def get_ranks(args, network_name, kernel_size_string=""):
    global x_attack, y_attack, metadata_profiling, metadata_attack, dk_plain, key_guesses
    (_, _), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(
        args.trace_file, load_metadata=True)
    key_guesses = util.load_csv('{}/ASCAD/key_guesses.csv'.format(args.traces_path),
                                delimiter=' ',
                                dtype=np.int,
                                start=0,
                                size=args.attack_size)
    # Manipulate the data
    x_attack = x_attack[:args.attack_size]
    y_attack = y_attack[:args.attack_size]
    if args.unmask:
        if args.use_hw:
            y_attack = np.array([y_attack[i] ^ metadata_attack[i]['masks'][0] for i in range(len(y_attack))])
        else:
            y_attack = np.array([util.HW[y_attack[i] ^ metadata_attack[i]['masks'][0]] for i in range(len(y_attack))])
    real_key = metadata_attack[0]['key'][args.subkey_index]

    # Load additional plaintexts
    dk_plain = None
    if network_name in req_dk:
        dk_plain = metadata_attack[:]['plaintext'][:, args.subkey_index]
        dk_plain = hot_encode(dk_plain, 9 if args.use_hw else 256, dtype=np.float)

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
    util.save_np('{}/model_r{}_{}{}.exp'.format(folder, run, network_name, kernel_size_string), y, f="%f")


def run_load(args):
    # traces_path = '/media/rico/Data/TU/thesis/data/'
    # models_path = '/media/rico/Data/TU/thesis/runs2/'
    # traces_path = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/student-datasets/'
    # models_path = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/rtubbing/'

    args.type_network = 'HW' if args.use_hw else 'ID'

    args.trace_file = '{}/ASCAD/ASCAD_{}_desync{}.h5'.format(args.traces_path, args.subkey_index, args.desync)
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
