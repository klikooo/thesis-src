from decimal import Decimal
from multiprocessing import Process

import torch

import util
import numpy as np

from models.load_model import load_model
from util import load_ascad, shuffle_permutation, DataSet, req_dk, hot_encode
from test import accuracy, test_with_key_guess_p

traces_path = '/media/rico/Data/TU/thesis/data/'
models_path = '/media/rico/Data/TU/thesis/runs2/'
# traces_path = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/student-datasets/'
# models_path = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/rtubbing/'

#####################################################################################
# Parameters
use_hw = False
n_classes = 9 if use_hw else 256
spread_factor = 1
runs = [x for x in range(5)]
train_size = 40000
epochs = 150
batch_size = 100
lr = 0.0001
sub_key_index = 2
attack_size = 3000
rank_step = 1
type_network = 'HW' if use_hw else 'ID'
unmask = True  # False if sub_key_index < 2 else True
data_set = DataSet.ASCAD
kernel_sizes = [3, 5, 7, 9, 11, 13, 15]

# network_names = ['SpreadV2', 'SpreadNet', 'DenseSpreadNet', 'MLPBEST']
network_names = ['ConvNetKernelAscad']
plt_titles = ['$Spread_{PH}$', '$Dense_{RT}$', '$MLP_{best}$', '', '', '', '']
only_accuracy = False
desync = 100
num_exps = 100
#####################################################################################

if len(plt_titles) != len(network_names):
    plt_titles = network_names

trace_file = '{}/ASCAD/ASCAD_{}_desync{}.h5'.format(traces_path, sub_key_index, desync)
device = torch.device("cuda")

permutations = util.generate_permutations(num_exps, attack_size)


def get_ranks(network_name, kernel_size_string=""):
    global x_attack, y_attack, metadata_profiling, metadata_attack, dk_plain, key_guesses
    (_, _), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(trace_file, load_metadata=True)
    key_guesses = util.load_csv('{}/ASCAD/key_guesses.csv'.format(traces_path),
                                delimiter=' ',
                                dtype=np.int,
                                start=0,
                                size=attack_size)
    # Manipulate the data
    x_attack = x_attack[:attack_size]
    y_attack = y_attack[:attack_size]
    if unmask:
        if use_hw:
            y_attack = np.array([y_attack[i] ^ metadata_attack[i]['masks'][0] for i in range(len(y_attack))])
        else:
            y_attack = np.array([util.HW[y_attack[i] ^ metadata_attack[i]['masks'][0]] for i in range(len(y_attack))])
    real_key = metadata_attack[0]['key'][sub_key_index]

    # Load additional plaintexts
    dk_plain = None
    if network_name in req_dk:
        dk_plain = metadata_attack[:]['plaintext'][:, sub_key_index]
        dk_plain = hot_encode(dk_plain, 9 if use_hw else 256, dtype=np.float)

    folder = '{}/{}/subkey_{}/{}{}{}_SF{}_' \
             'E{}_BZ{}_LR{}/train{}/'.format(
                models_path,
                str(data_set),
                sub_key_index,
                '' if unmask else 'masked/',
                '' if desync is 0 else 'desync{}/'.format(desync),
                type_network,
                spread_factor,
                epochs,
                batch_size,
                '%.2E' % Decimal(lr),
                train_size)

    # Calculate the predictions before hand
    predictions = []
    for run in runs:
        model_path = '{}/model_r{}_{}{}.pt'.format(
            folder,
            run,
            network_name,
            kernel_size_string)
        print('path={}'.format(model_path))

        model = load_model(network_name, model_path)
        model.eval()
        print("Using {}".format(model))
        model.to(device)

        # Calculate predictions
        prediction = accuracy(model, x_attack, y_attack, dk_plain)
        predictions.append(prediction.cpu().numpy())

    # Start a thread for each run
    processes = []
    for i, run in enumerate(runs):
        p = Process(target=threaded_run_test, args=(predictions[i], folder, run,
                                                    network_name, kernel_size_string, real_key))
        processes.append(p)
        p.start()
    # Wait for them to finish
    for p in processes:
        p.join()
        print('Joined process')


def threaded_run_test(prediction, folder, run, network_name, kernel_size_string, real_key):
    # Shuffle the data using same permutation  for n_exp and calculate mean for GE of the model
    y = []
    for exp_i in range(num_exps):
        # Select permutation
        permutation = permutations[exp_i]

        # Shuffle data
        predictions_shuffled = shuffle_permutation(permutation, np.array(prediction))
        key_guesses_shuffled = shuffle_permutation(permutation, key_guesses)

        # Test the data
        x_exp, y_exp = test_with_key_guess_p(key_guesses_shuffled, predictions_shuffled,
                                             attack_size=attack_size,
                                             real_key=real_key,
                                             use_hw=use_hw)
        y.append(y_exp)

    # Calculate the mean over the experiments
    y = np.mean(y, axis=0)
    util.save_np('{}/model_r{}_{}{}.exp'.format(folder, run, network_name, kernel_size_string), y, f="%f")


# Test the networks that were specified
for net_name in network_names:
    if net_name in util.req_kernel_size:
        for kernel_size in kernel_sizes:
            kernel_string = "_k{}".format(kernel_size)

            get_ranks(net_name, kernel_string)
    else:
        get_ranks(net_name)
