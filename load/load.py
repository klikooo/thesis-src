import math
from decimal import Decimal

import torch

import util
from models.SpreadNetIn import SpreadNetIn
import numpy as np

import matplotlib.pyplot as plt

from models.SpreadV2 import SpreadV2
from models.load_model import load_model
from util import load_ascad, shuffle_permutation, DataSet, req_dk, hot_encode, SBOX
from test import test, test_with_key_guess, accuracy, test_with_key_guess_p

path = '/media/rico/Data/TU/thesis'

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
desync = 0
num_exps = 50
#####################################################################################

if len(plt_titles) != len(network_names):
    plt_titles = network_names

trace_file = '{}/data/ASCAD/ASCAD_{}_desync{}.h5'.format(path, sub_key_index, desync)
device = torch.device("cuda")


permutations = util.generate_permutations(num_exps, attack_size)


def get_ranks(use_hw, runs, train_size,
              epochs, lr, sub_key_index, attack_size, rank_step, unmask, network_name,
              kernel_size_string=""):
    ranks_x = []
    ranks_y = []
    (_, _), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(trace_file, load_metadata=True)
    key_guesses = util.load_csv('/media/rico/Data/TU/thesis/data/ASCAD/key_guesses.csv',
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
        folder = '/media/rico/Data/TU/thesis/runs2/{}/subkey_{}/{}{}{}_SF{}_' \
                     'E{}_BZ{}_LR{}/train{}/'.format(
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

        # Calculate the mean over the experimentfs
        y = np.mean(y, axis=0)
        util.save_np('{}/model_r{}_{}{}.exp'.format(folder, run, network_name, kernel_size_string), y, f="%f")

        if isinstance(model, SpreadNetIn):
            # Get the intermediate values right after the first fully connected layer
            z = np.transpose(model.intermediate_values2[0])

            # Calculate the mse for the maximum and minimum from these traces and the learned min and max
            min_z = np.min(z, axis=1)
            max_z = np.max(z, axis=1)
            msq_min = np.mean(np.square(min_z - model.tensor_min), axis=None)
            msq_max = np.mean(np.square(max_z - model.tensor_max), axis=None)
            print('msq min: {}'.format(msq_min))
            print('msq max: {}'.format(msq_max))

            # Plot the distribution of each neuron right after the first fully connected layer
            for k in [50]:
                plt.grid(True)
                plt.axvline(x=model.tensor_min[k], color='green')
                plt.axvline(x=model.tensor_max[k], color='green')
                plt.hist(z[:][k], bins=40)

                plt.show()
            exit()

            # Retrieve the intermediate values right after the spread layer,
            # and order them such that each 6 values after each other belong to the neuron of the
            # previous layer
            v = model.intermediate_values
            order = [int((x % spread_factor) * 100 + math.floor(x / spread_factor)) for x in range(spread_factor * 100)]
            inter = []
            for x in range(len(v[0])):
                inter.append([v[0][x][j] for j in order])

            # Calculate the standard deviation of each neuron in the spread layer
            std = np.std(inter, axis=0)
            threshold = 1.0 / attack_size * 10
            print("divby: {}".format(threshold))
            res = np.where(std < threshold, 1, 0)

            # Calculate the mean of each neuron in the spread layer
            mean_res = np.mean(inter, axis=0)
            # mean_res2 = np.where(mean_res < threshold, 1, 0)
            mean_res2 = np.where(mean_res == 0.0, 1, 0)
            print('Sum  std results {}'.format(np.sum(res)))
            print('Sum mean results {}'.format(np.sum(mean_res2)))

            # Check which neurons have a std and mean where it is smaller than threshold
            total_same = 0
            for j in range(len(mean_res2)):
                if mean_res2[j] == 1 and res[j] == 1:
                    total_same += 1
            print('Total same: {}'.format(total_same))

            # Plot the standard deviations
            plt.title('Comparison of networks')
            plt.xlabel('#neuron')
            plt.ylabel('std')
            xcoords = [j * spread_factor for j in range(100)]
            for xc in xcoords:
                plt.axvline(x=xc, color='green')
            plt.grid(True)
            plt.plot(std, label='std')
            plt.figure()

            # Plot the means
            plt.title('Performance of networks')
            plt.xlabel('#neuron')
            plt.ylabel('mean')
            for xc in xcoords:
                plt.axvline(x=xc, color='green')
            plt.grid(True)
            plt.plot(mean_res, label='mean')
            plt.legend()
            plt.show()

        ranks_x.append(x)
        ranks_y.append(y)
    return ranks_x, ranks_y


# Test the networks that were specified
ranks_x = []
ranks_y = []
rank_mean_y = []
name_models = []
for network_name in network_names:
    if network_name in util.req_kernel_size:
        for kernel_size in kernel_sizes:
            kernel_string = "_k{}".format(kernel_size)

            x, y = get_ranks(use_hw, runs, train_size, epochs, lr, sub_key_index,
                             attack_size, rank_step, unmask, network_name, kernel_string)
            mean_y = np.mean(y, axis=0)
            ranks_x.append(x)
            ranks_y.append(y)
            rank_mean_y.append(mean_y)
            name_models.append("{} K{}".format(network_name, kernel_size))
    else:
        x, y = get_ranks(use_hw, runs, train_size, epochs, lr, sub_key_index,
                         attack_size, rank_step, unmask, network_name)
        mean_y = np.mean(y, axis=0)
        ranks_x.append(x)
        ranks_y.append(y)
        rank_mean_y.append(mean_y)
        name_models.append(network_name)

for i in range(len(rank_mean_y)):
    plt.title('Performance of {}'.format(name_models[i]))
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)

    # Plot the results
    for x, y in zip(ranks_x[i], ranks_y[i]):
        plt.plot(x, y)
    figure = plt.gcf()
    plt.figure()
    figure.savefig('/home/rico/Pictures/{}.png'.format(name_models[i]), dpi=100)

# plt.title('Comparison of networks')
plt.xlabel('Number of traces')
plt.ylabel('Mean rank')
plt.grid(True)
for i in range(len(rank_mean_y)):
    plt.plot(ranks_x[i][0], rank_mean_y[i], label=name_models[i])
    plt.legend()

    # plt.figure()
figure = plt.gcf()
figure.savefig('/home/rico/Pictures/{}.png'.format('mean'), dpi=100)

plt.show()
