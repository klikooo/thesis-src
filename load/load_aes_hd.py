import math
from decimal import Decimal

import torch

from models import DenseSpreadNet
from models.CosNet import CosNet
from models.DenseNet import DenseNet
from models.SpreadNet import SpreadNet
from models.SpreadNetIn import SpreadNetIn
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.SpreadV2 import SpreadV2
from util import load_ascad, shuffle_permutation, load_csv, load_aes_hd, load_dpav4, DataSet
from test import test, test_with_key_guess

path = '/media/rico/Data/TU/thesis'

#####################################################################################
# Parameters
use_hw = False
n_classes = 9 if use_hw else 256
spread_factor = 6
runs = [x for x in range(5)]
train_size = 10000
epochs = 80
batch_size = 100
lr = 0.001
sub_key_index = 2
attack_size = 10000
rank_step = 1
type_network = 'HW' if use_hw else 'ID'
unmask = False if sub_key_index < 2 else True

# network_names = ['SpreadV2', 'SpreadNet', 'DenseSpreadNet', 'MLPBEST']
network_names = ['DenseSpreadNet', 'SpreadV2', 'SpreadNet']
plt_titles = ['$Spread_{PH}$', '$Dense_{RT}$', '$MLP_{best}$', '']
only_accuracy = False
data_set = DataSet.AES_HD
#####################################################################################
data_set_name = str(data_set)
if len(plt_titles) != len(network_names):
    plt_titles = network_names


device = torch.device("cuda")


# Load Data
x_attack, y_attack = load_dpav4({'use_hw': use_hw,
                                  'traces_path': '/media/rico/Data/TU/thesis/data'})
key_guesses = np.transpose(
    load_csv('/media/rico/Data/TU/thesis/data/{}/Value/key_guesses_ALL.csv'.format(data_set_name), delimiter=' ', dtype=np.int))

x_attack = x_attack[train_size:train_size + attack_size]
y_attack = y_attack[train_size:train_size + attack_size]
key_guesses = key_guesses[train_size:train_size + attack_size]


def get_ranks(x_attack, y_attack, key_guesses, runs, train_size,
              epochs, lr, sub_key_index, attack_size, rank_step, unmask, network_name):
    ranks_x = []
    ranks_y = []

    for run in runs:
        model_path = '/media/rico/Data/TU/thesis/runs/' \
                     '{}/subkey_{}/{}_SF{}_E{}_BZ{}_LR{}/train{}/model_r{}_{}.pt'.format(
            data_set_name,
            sub_key_index,
            type_network,
            spread_factor,
            epochs,
            batch_size,
            '%.2E' % Decimal(lr),
            train_size,
            run,
            network_name
        )
        print('path={}'.format(model_path))

        if "DenseSpreadNet" in network_name:
            model = DenseSpreadNet.DenseSpreadNet.load_model(model_path)
        elif "MLP" in network_name:
            model = DenseNet.load_model(model_path)
        elif "SpreadV2" in network_name:
            model = SpreadV2.load_spread(model_path)
        # elif "SpreadNet" in network_name:
        #     model = SpreadNetIn.load_spread(model_path)
        elif "SpreadNet" in network_name:
            model = SpreadNet.load_spread(model_path)
        elif "CosNet" in network_name:
            model = CosNet.load_model(model_path)
        else:
            raise Exception("Unknown model")
        print("Using {}".format(model))
        model.to(device)

        # permutation = np.random.permutation(x_attack.shape[0])
        # x_attack = shuffle_permutation(permutation, np.array(x_attack))
        # y_attack = shuffle_permutation(permutation, np.array(y_attack))

        x, y = test_with_key_guess(x_attack, y_attack, key_guesses, model, attack_size, n_classes=256)
        ranks_x.append(x)
        ranks_y.append(y)

        # accuracy()
        # data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
        # print('x_test size: {}'.format(data.cpu().size()))
        # predictions = F.softmax(model(data).to(device), dim=-1).to(device)
    return ranks_x, ranks_y


ranks_x = []
ranks_y = []
rank_mean_y = []
for network_name in network_names:
    x, y = get_ranks(x_attack, y_attack, key_guesses, runs, train_size, epochs, lr, sub_key_index,
                     attack_size, rank_step, unmask, network_name)
    mean_y = np.mean(y, axis=0)
    ranks_x.append(x)
    ranks_y.append(y)
    rank_mean_y.append(mean_y)


for i in range(len(rank_mean_y)):
    plt.title('Performance of {}'.format(plt_titles[i]))
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)

    # Plot the results
    for x, y in zip(ranks_x[i], ranks_y[i]):
        plt.plot(x, y)
    figure = plt.gcf()
    plt.figure()
    figure.savefig('/home/rico/Pictures/{}.png'.format(network_names[i]), dpi=100)


# plt.title('Comparison of networks')
plt.xlabel('Number of traces')
plt.ylabel('Mean rank')
plt.grid(True)
for i in range(len(rank_mean_y)):
    plt.plot(ranks_x[i][0], rank_mean_y[i], label=plt_titles[i])
    plt.legend()

    # plt.figure()
figure = plt.gcf()
figure.savefig('/home/rico/Pictures/{}.png'.format('mean'), dpi=100)

plt.show()
