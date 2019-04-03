import itertools
from decimal import Decimal

import util
import numpy as np

import matplotlib.pyplot as plt

from util_classes import get_save_name

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
lr = 0.0001
sub_key_index = 2
rank_step = 1
type_network = 'HW' if use_hw else 'ID'
unmask = True  # False if sub_key_index < 2 else True
data_set = util.DataSet.RANDOM_DELAY
kernel_sizes = [20, 30, 40, 50, 70, 70, 80, 90, 100, 110]
channel_sizes = [8]
num_layers = []

# network_names = ['SpreadV2', 'SpreadNet', 'DenseSpreadNet', 'MLPBEST']
network_names = [ 'ConvNetKernelSmall']
plt_titles = ['$Spread_{PH}$', '$Dense_{RT}$', '$MLP_{best}$', '', '', '', '']
only_accuracy = False
desync = 0
show_losses = False
show_acc = False
experiment = False
#####################################################################################


def get_ge(net_name, model_parameters):
    folder = '/media/rico/Data/TU/thesis/runs{}/{}/subkey_{}/{}{}{}_SF{}_' \
             'E{}_BZ{}_LR{}/train{}/'.format(
                                    '2' if not experiment else '',
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
    lta, lva, ltl, lvl = [], [], [], []
    for run in runs:
        filename = '{}/model_r{}_{}'.format(
            folder,
            run,
            get_save_name(net_name, model_parameters))
        ge_path = '{}.exp'.format(filename)

        y_r = util.load_csv(ge_path, delimiter=' ', dtype=np.float)
        x_r = range(len(y_r))
        ge_x.append(x_r)
        ge_y.append(y_r)

        if show_losses or show_acc:
            ta, va, tl, vl = util.load_loss_acc(filename)
            lta.append(ta)
            lva.append(va)
            ltl.append(tl)
            lvl.append(vl)

    return ge_x, ge_y, (lta, lva, ltl, lvl)


# Test the networks that were specified
ranks_x = []
ranks_y = []
rank_mean_y = []
name_models = []
model_params = {}
all_loss_acc = []  # ([], [], [], [])
for network_name in network_names:
    def lambda_kernel(x): model_params.update({"kernel_size": x})


    def lambda_channel(x): model_params.update({"channel_size": x})

    def lambda_layers(x): model_params.update({"num_layers": x})

    def retrieve_ge():
        print(model_params)
        ge_x, ge_y, loss_acc = get_ge(network_name, model_params)
        mean_y = np.mean(ge_y, axis=0)
        ranks_x.append(ge_x)
        ranks_y.append(ge_y)
        rank_mean_y.append(mean_y)
        name_models.append(get_save_name(network_name, model_params))

        all_loss_acc.append(loss_acc)

    util.loop_at_least_once(kernel_sizes, lambda_kernel, lambda: (
        util.loop_at_least_once(channel_sizes, lambda_channel, lambda: (
            util.loop_at_least_once(num_layers, lambda_layers, retrieve_ge)
        ))
    ))

line_marker = itertools.cycle(('+', '.', 'o', '*'))
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
    plt.plot(ranks_x[i][0], rank_mean_y[i], label=name_models[i], marker=next(line_marker))
    plt.legend()

    # plt.figure()
figure = plt.gcf()
figure.savefig('/home/rico/Pictures/{}.png'.format('mean'), dpi=100)


if show_losses or show_acc:
    for i in range(len(runs)):
        (acc_train, acc_vali, loss_train, loss_vali) = all_loss_acc[i]

        for run in range(len(runs)):
            plt.figure()
            plt.title('Accuracy during training {}'.format(name_models[i]))
            plt.xlabel('Accuracy')
            plt.ylabel('Epoch')
            plt.grid(True)
            # Plot the accuracy
            # for x, y in zip(ranks_x[i], ranks_y[i]):
            plt.plot([x for x in range(len(acc_train[run]))], acc_train[run], label="Train")
            plt.plot([x for x in range(len(acc_train[run]))], acc_vali[run], label="Vali")
            plt.legend()
            plt.plot(acc_vali)

            plt.figure()
            plt.title('Loss during training {}'.format(name_models[i]))
            plt.xlabel('Loss')
            plt.ylabel('Epoch')
            plt.grid(True)
            # Plot the accuracy
            # for x, y in zip(ranks_x[i], ranks_y[i]):
            plt.plot([x for x in range(len(loss_train[run]))], loss_train[run], label="Train")
            plt.plot([x for x in range(len(loss_train[run]))], loss_vali[run], label="Vali")
            plt.legend()
            plt.plot(acc_vali)


plt.show()
