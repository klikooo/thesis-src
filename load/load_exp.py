from decimal import Decimal

import util
import numpy as np

import matplotlib.pyplot as plt

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
attack_size = 500
rank_step = 1
type_network = 'HW' if use_hw else 'ID'
unmask = True  # False if sub_key_index < 2 else True
data_set = util.DataSet.ASCAD
kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

# network_names = ['SpreadV2', 'SpreadNet', 'DenseSpreadNet', 'MLPBEST']
network_names = ['ConvNetKernelAscad']
plt_titles = ['$Spread_{PH}$', '$Dense_{RT}$', '$MLP_{best}$', '', '', '', '']
only_accuracy = False
desync = 0
#####################################################################################


def get_ge(net_name, kernel_size_string=""):
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

    ge_x, ge_y = [], []
    for run in runs:
        ge_path = '{}/model_r{}_{}{}.exp'.format(
            folder,
            run,
            net_name,
            kernel_size_string)

        y_r = util.load_csv(ge_path, delimiter=' ', dtype=np.float)
        x_r = range(len(y_r))
        ge_x.append(x_r)
        ge_y.append(y_r)

    return ge_x, ge_y


# Test the networks that were specified
ranks_x = []
ranks_y = []
rank_mean_y = []
name_models = []
for network_name in network_names:
    if network_name in util.req_kernel_size:
        for kernel_size in kernel_sizes:
            kernel_string = "_k{}".format(kernel_size)

            x, y = get_ge(network_name, kernel_string)
            mean_y = np.mean(y, axis=0)
            ranks_x.append(x)
            ranks_y.append(y)
            rank_mean_y.append(mean_y)
            name_models.append("{} K{}".format(network_name, kernel_size))
    else:
        x, y = get_ge(network_name)
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
