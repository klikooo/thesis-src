from decimal import Decimal

import torch

import numpy as np

import matplotlib.pyplot as plt

from models.load_model import load_model
from test import test_with_key_guess
import util

path = '/media/rico/Data/TU/thesis'

#####################################################################################
# Parameters
# use_hw = True
# n_classes = 9 if use_hw else 256
# spread_factor = 3
# runs = [x for x in range(10)]
# train_size = 45000
# epochs = 80
# batch_size = 1000
# lr = 0.001
# sub_key_index = 2
# attack_size = 5000
# rank_step = 1
# type_network = 'HW' if use_hw else 'ID'
# unmask = False if sub_key_index < 2 else True

# network_names = ['SpreadV2', 'SpreadNet']
# network_names = ['SpreadV2', 'SpreadNet', 'DenseSpreadNet', 'MLPBEST']
# only_accuracy = False
# data_set = util.DataSet.RANDOM_DELAY
#####################################################################################
# data_set_name = str(data_set)
# if len(plt_titles) != len(network_names):
#     plt_titles = network_names
device = torch.device("cuda")


def get_ranks(x_attack, y_attack, key_guesses, real_key, runs, train_size,
              epochs, lr, sub_key_index, attack_size, network_name, batch_size, spread_factor, use_hw, data_set_name):
    ranks_x = []
    ranks_y = []

    for run in runs:
        model_path = '/media/rico/Data/TU/thesis/runs2/' \
                     '{}/subkey_{}/{}_SF{}_E{}_BZ{}_LR{}/train{}/model_r{}_{}.pt'.format(
                        data_set_name,
                        sub_key_index,
                        'HW' if use_hw else 'ID',
                        spread_factor,
                        epochs,
                        batch_size,
                        '%.2E' % Decimal(lr),
                        train_size,
                        run,
                        network_name
                        )
        print('path={}'.format(model_path))

        # Load the model
        model = load_model(network_name=network_name, model_path=model_path)
        print("Using {}".format(model))
        model.to(device)

        # permutation = np.random.permutation(x_attack.shape[0])
        # x_attack = shuffle_permutation(permutation, np.array(x_attack))
        # y_attack = shuffle_permutation(permutation, np.array(y_attack))

        x, y = test_with_key_guess(x_attack, y_attack, key_guesses, model,
                                   attack_size=attack_size,
                                   real_key=real_key,
                                   use_hw=use_hw)
        # Add the ranks
        ranks_x.append(x)
        ranks_y.append(y)
    return ranks_x, ranks_y


def main(use_hw):
    data_set = util.DataSet.RANDOM_DELAY
    total_x_attack, total_y_attack, total_key_guesses, real_key = load_all_data(use_hw, data_set)

    train_sizes = [45000]
    batch_sizes = [100, 200, 500, 1000]
    attack_sizes = [5000]
    learning_rates = [0.01, 0.001, 0.0001]
    network_names = ['SpreadV2', 'SpreadNet', 'DenseSpreadNet', 'MLPBEST']
    spread_factors = [3, 6, 9]
    for train_size in train_sizes:
        for attack_size in attack_sizes:
            x_attack = total_x_attack[train_size:train_size + attack_size]
            y_attack = total_y_attack[train_size:train_size + attack_size]
            key_guesses = total_key_guesses[train_size:train_size + attack_size]
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for spread_factor in spread_factors:

                        create_picture(x_attack, y_attack, key_guesses, real_key,
                                       train_size=train_size,
                                       learning_rate=learning_rate,
                                       attack_size=attack_size,
                                       network_names=network_names,
                                       use_hw=use_hw,
                                       batch_size=batch_size,
                                       spread_factor=spread_factor,
                                       data_set=data_set)


def load_all_data(use_hw, data_set):
    # Load Data
    loader = util.load_data_set(data_set)
    data_set_name = str(data_set)
    total_x_attack, total_y_attack = loader({'use_hw': use_hw,
                                             'traces_path': '/media/rico/Data/TU/thesis/data'})
    total_key_guesses = np.transpose(
        util.load_csv('/media/rico/Data/TU/thesis/data/{}/Value/key_guesses_ALL.csv'.format(data_set_name),
                      delimiter=' ',
                      dtype=np.int))
    real_key = util.load_csv('/media/rico/Data/TU/thesis/data/{}/secret_key.csv'.format(data_set_name), dtype=np.int)
    return total_x_attack, total_y_attack, total_key_guesses, real_key


def create_picture(x_attack, y_attack, key_guesses, real_key,
                   train_size,
                   learning_rate,
                   attack_size,
                   network_names,
                   use_hw,
                   batch_size,
                   spread_factor,
                   data_set):

    ranks_x = []
    ranks_y = []
    rank_mean_y = []
    for network_name in network_names:
        x, y = get_ranks(x_attack, y_attack, key_guesses, real_key,
                         runs=range(10),
                         train_size=train_size,
                         epochs=80,
                         lr=learning_rate,
                         sub_key_index=2,
                         attack_size=attack_size,
                         network_name=network_name,
                         use_hw=use_hw,
                         batch_size=batch_size,
                         spread_factor=spread_factor,
                         data_set_name=str(data_set)
                         )
        mean_y = np.mean(y, axis=0)
        ranks_x.append(x)
        ranks_y.append(y)
        rank_mean_y.append(mean_y)

    plt_titles = ['$Spread_{V2}$', '$Spread_{PH}$', '$Dense_{RT}$', '$MLP_{best}$']

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
        save_fig_path = '/media/rico/Data/TU/thesis/pictures/' \
                        '{}/{}_{}_SF{}_BZ{}_LR{}_train{}.png'.format(
                            str(data_set),
                            network_names[i],
                            'HW' if use_hw else 'ID',
                            spread_factor,
                            batch_size,
                            '%.2E' % Decimal(learning_rate),
                            train_size)
        figure.savefig(save_fig_path, dpi=100)

    # plt.title('Comparison of networks')
    plt.xlabel('Number of traces')
    plt.ylabel('Mean rank')
    plt.grid(True)
    for i in range(len(rank_mean_y)):
        plt.plot(ranks_x[i][0], rank_mean_y[i], label=plt_titles[i])
        plt.legend()

        # plt.figure()
    figure = plt.gcf()
    save_fig_path = '/media/rico/Data/TU/thesis/pictures/' \
                    '{}/{}_{}_SF{}_BZ{}_LR{}_train{}.png'.format(
                        str(data_set),
                        'MEAN',
                        'HW' if use_hw else 'ID',
                        spread_factor,
                        batch_size,
                        '%.2E' % Decimal(learning_rate),
                        train_size)
    figure.savefig(save_fig_path, dpi=100)

    # plt.show()


main(False)
