from decimal import Decimal

import torch

import numpy as np

import matplotlib.pyplot as plt

from models.load_model import load_model
from test import test_with_key_guess
import util
import pdb

path = '/media/rico/Data/TU/thesis'

#####################################################################################
# Parameters
use_hw = False
n_classes = 9 if use_hw else 256
spread_factor = 1
runs = [x for x in range(5)]
train_size = 20000
epochs = 120
batch_size = 100
lr = 0.0005
sub_key_index = 2
attack_size = 100
rank_step = 1
type_network = 'HW' if use_hw else 'ID'
unmask = False if sub_key_index < 2 else True

# network_names = ['SpreadV2', 'SpreadNet']
network_names = ['ConvNetKernel']
kernel_sizes = [3, 5, 7, 9, 11, 13, 15]
# network_names = ['ConvNet', 'ConvNetDK']
plt_titles = ['$Spread_{V2}$', '$Spread_{PH}$', '$Dense_{RT}$', '$MLP_{best}$']
only_accuracy = False
data_set = util.DataSet.RANDOM_DELAY
raw_traces = True
validation_size = 1000

#####################################################################################

data_set_name = str(data_set)
if len(plt_titles) != len(network_names):
    plt_titles = network_names
device = torch.device("cuda")

# Load Data
loader = util.load_data_set(data_set)

print('Loading data set')
total_x_attack, total_y_attack, plain = loader({'use_hw': use_hw,
                                                'traces_path': '/media/rico/Data/TU/thesis/data',
                                                'raw_traces': raw_traces,
                                                'start': train_size + validation_size,
                                                'size': attack_size,
                                                'domain_knowledge': True})
print('Loading key guesses')
key_guesses = util.load_csv('/media/rico/Data/TU/thesis/data/{}/Value/key_guesses_ALL_transposed.csv'.format(
    data_set_name),
    delimiter=' ',
    dtype=np.int,
    start=train_size + validation_size,
    size=attack_size)

real_key = util.load_csv('/media/rico/Data/TU/thesis/data/{}/secret_key.csv'.format(data_set_name), dtype=np.int)

x_attack = total_x_attack
y_attack = total_y_attack


# permutation = np.random.permutation(x_attack.shape[0])
# permutation = np.arange(0, x_attack.shape[0])


def get_ranks(x_attack, y_attack, key_guesses, runs, train_size,
              epochs, lr, sub_key_index, attack_size, rank_step, unmask, network_name, kernel_size_string=""):
    ranks_x = []
    ranks_y = []

    for run in runs:
        model_path = '/media/rico/Data/TU/thesis/runs2/' \
                     '{}/subkey_{}/{}_SF{}_E{}_BZ{}_LR{}/train{}/model_r{}_{}{}.pt'.format(
                        data_set_name,
                        sub_key_index,
                        type_network,
                        spread_factor,
                        epochs,
                        batch_size,
                        '%.2E' % Decimal(lr),
                        train_size,
                        run,
                        network_name,
                        kernel_size_string)
        print('path={}'.format(model_path))

        # Load the model
        model = load_model(network_name=network_name, model_path=model_path)
        model.eval()
        print("Using {}".format(model))
        model.to(device)

        # Number of times we test a single model + shuffle the test traces
        num_exps = 100
        x, y = [], []
        for exp_i in range(num_exps):
            permutation = np.random.permutation(x_attack.shape[0])
            # permutation = np.arange(0, x_attack.shape[0])

            x_attack_shuffled = util.shuffle_permutation(permutation, np.array(x_attack))
            y_attack_shuffled = util.shuffle_permutation(permutation, np.array(y_attack))
            key_guesses_shuffled = util.shuffle_permutation(permutation, key_guesses)

            # Check if we need domain knowledge
            dk_plain = None
            if network_name in util.req_dk:
                dk_plain = plain
                dk_plain = util.shuffle_permutation(permutation, dk_plain)

            x_exp, y_exp = test_with_key_guess(x_attack_shuffled, y_attack_shuffled, key_guesses_shuffled, model,
                                               attack_size=attack_size,
                                               real_key=real_key,
                                               use_hw=use_hw,
                                               plain=dk_plain)
            x = x_exp
            y.append(y_exp)

        # Take the mean of the different experiments
        y = np.mean(y, axis=0)
        # Add the ranks
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

            x, y = get_ranks(x_attack, y_attack, key_guesses, runs, train_size, epochs, lr, sub_key_index,
                             attack_size, rank_step, unmask, network_name, kernel_string)
            mean_y = np.mean(y, axis=0)
            ranks_x.append(x)
            ranks_y.append(y)
            rank_mean_y.append(mean_y)
            name_models.append("{} K{}".format(network_name, kernel_size))
    else:
        x, y = get_ranks(x_attack, y_attack, key_guesses, runs, train_size, epochs, lr, sub_key_index,
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
