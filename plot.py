import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def compare(x, ranks_spread, ranks_dense):
    rank_avg_spread = np.mean(ranks_spread, axis=0)
    rank_avg_dense = np.mean(ranks_dense, axis=0)

    # Mean figure
    # plt.title('Performance of {}'.format('a'))
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(x[0], rank_avg_spread, label='Spread')
    plt.plot(x[0], rank_avg_dense, label='Dense')
    plt.legend()
    plt.show()
    plt.figure()


def compare1(ranks_x, ranks_y, titles):
    # Mean figure
    # plt.title('Performance of {}'.format('a'))
    plt.xlabel('Number of traces')
    plt.ylabel('Mean rank')
    plt.grid(True)
    for i in range(len(ranks_y)):
        rank_avg = np.mean(ranks_y[i], axis=0)
        plt.plot(ranks_x[i][0], rank_avg, label=titles[i])
    plt.legend()
    plt.show()
    plt.figure()


def get(path, file):
    with open('{}/{}'.format(path, file), 'rb') as f:
        ret = pickle.load(f)
    return ret


if __name__ == '__main__':
    use_hw = False
    type_network = 'HW' if use_hw else 'ID'
    train_size = 50000
    batch_size = 100
    subkey_index = 2
    epochs = 500
    path = '/media/rico/Data/TU/thesis/runs/subkey_{}/{}_SF6_E{}_BZ{}_LR1E-5/train{}'.format(
        subkey_index, type_network, epochs, batch_size, train_size)

    os.chdir(path)
    names = []
    for file in glob.glob("*.r"):
        if file.startswith("y_"):
            names.append(file[:-2][2:])

    print(names)
    #
    # dense_file = 'Dense-spread network'
    # spread_file = 'Spread network'
    #
    # x_save_file = '{}/x_{}.r'.format(path, dense_file)
    # dense_save_file = '{}/y_{}.r'.format(path, dense_file)
    # spread_save_file = '{}/y_{}.r'.format(path, spread_file)
    #
    # with open(x_save_file, 'rb') as f:
    #     ranks_x = pickle.load(f)
    #
    # with open(dense_save_file, 'rb') as f:
    #     ranks_dense = pickle.load(f)
    # with open(spread_save_file, 'rb') as f:
    #     ranks_spread = pickle.load(f)

    # compare(ranks_x, ranks_spread, ranks_dense)

    ranks_x = []
    ranks_y = []
    for name in names:
        ranks_x.append(get(path, 'x_{}.r'.format(name)))
        ranks_y.append(get(path, 'y_{}.r'.format(name)))

    compare1(ranks_x, ranks_y, names)


