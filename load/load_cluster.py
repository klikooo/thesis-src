from decimal import Decimal

import torch

import util
import numpy as np

from models.load_model import load_model
from util import load_ascad, shuffle_permutation, DataSet, req_dk, hot_encode
from test import accuracy, test_with_key_guess_p

traces_path = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/student-datasets/'
models_path = '/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/rtubbing/'

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
attack_size = 2000
rank_step = 1
type_network = 'HW' if use_hw else 'ID'
unmask = True  # False if sub_key_index < 2 else True
data_set = DataSet.ASCAD
kernel_sizes = [3, 5, 7, 9, 11, 13, 15]

# network_names = ['SpreadV2', 'SpreadNet', 'DenseSpreadNet', 'MLPBEST']
network_names = ['ConvNetKernelAscad']
plt_titles = ['$Spread_{PH}$', '$Dense_{RT}$', '$MLP_{best}$', '', '', '', '']
only_accuracy = False
desync = 50
num_exps = 50
#####################################################################################

if len(plt_titles) != len(network_names):
    plt_titles = network_names

trace_file = '{}/data/ASCAD/ASCAD_{}_desync{}.h5'.format(traces_path, sub_key_index, desync)
device = torch.device("cuda")

permutations = util.generate_permutations(num_exps, attack_size)


def get_ranks(use_hw, runs, train_size,
              epochs, lr, sub_key_index, attack_size, rank_step, unmask, network_name,
              kernel_size_string=""):
    ranks_x = []
    ranks_y = []
    (_, _), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(trace_file, load_metadata=True)
    key_guesses = util.load_csv('{}/ASCAD/key_guesses.csv'.format(traces_path),
                                delimiter=' ',
                                dtype=np.int,
                                start=0,
                                size=attack_size)
    x_attack = x_attack[:attack_size]
    y_attack = y_attack[:attack_size]
    if unmask:
        if use_hw:
            y_attack = np.array([y_attack[i] ^ metadata_attack[i]['masks'][0] for i in range(len(y_attack))])
        else:
            y_attack = np.array([util.HW[y_attack[i] ^ metadata_attack[i]['masks'][0]] for i in range(len(y_attack))])
    real_key = metadata_attack[0]['key'][sub_key_index]

    for run in runs:
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

        # Load additional plaintexts
        dk_plain = None
        if network_name in req_dk:
            dk_plain = metadata_attack[:]['plaintext'][:, sub_key_index]
            dk_plain = hot_encode(dk_plain, 9 if use_hw else 256, dtype=np.float)

        # Calculate predictions
        predictions = accuracy(model, x_attack, y_attack, dk_plain)
        predictions = predictions.cpu().numpy()

        # Shuffle the data using same permutation  for n_exp and calculate mean for GE of the model
        x, y = [], []
        for exp_i in range(num_exps):
            permutation = permutations[exp_i]

            # Shuffle data
            predictions_shuffled = shuffle_permutation(permutation, np.array(predictions))
            key_guesses_shuffled = shuffle_permutation(permutation, key_guesses)

            # Test the data
            x_exp, y_exp = test_with_key_guess_p(key_guesses_shuffled, predictions_shuffled,
                                                 attack_size=attack_size,
                                                 real_key=real_key,
                                                 use_hw=use_hw)
            x = x_exp
            y.append(y_exp)

        # Calculate the mean over the experiments
        y = np.mean(y, axis=0)
        util.save_np('{}/model_r{}_{}{}.exp'.format(folder, run, network_name, kernel_size_string), y, f="%f")

        ranks_x.append(x)
        ranks_y.append(y)
    return ranks_x, ranks_y


# Test the networks that were specified
for network_name in network_names:
    if network_name in util.req_kernel_size:
        for kernel_size in kernel_sizes:
            kernel_string = "_k{}".format(kernel_size)

            get_ranks(use_hw, runs, train_size, epochs, lr, sub_key_index,
                      attack_size, rank_step, unmask, network_name, kernel_string)
    else:
        get_ranks(use_hw, runs, train_size, epochs, lr, sub_key_index,
                  attack_size, rank_step, unmask, network_name)
