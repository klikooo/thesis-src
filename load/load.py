import math
from decimal import Decimal

import torch

from models import DenseSpreadNet
from models.DenseNet import DenseNet
from models.SpreadNet import SpreadNet
from models.SpreadNetIn import SpreadNetIn
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt


from util import load_ascad
from test import test


path = '/media/rico/Data/TU/thesis'

#####################################################################################
# Parameters
use_hw = True
n_classes = 9 if use_hw else 256
spread_factor = 3
runs = [x for x in range(10)]
train_size = 2000
epochs = 80
batch_size = 100
lr = 0.00001
sub_key_index = 2
attack_size = 3000
rank_step = 5
type_network = 'HW' if use_hw else 'ID'
unmask = False if sub_key_index < 2 else True
# network_name = 'SpreadNet'
# network_name = 'DenseSpreadNet'
# network_name = "MLPBEST"

network_names = ['SpreadNet', 'MLPBEST', 'DenseSpreadNet']
# network_names = ['SpreadNet']
# network_names = ['MLPBEST']
#####################################################################################

trace_file = '{}/data/ASCAD_{}.h5'.format(path, sub_key_index)
device = torch.device("cuda")


def get_ranks(use_hw, runs, train_size,
              epochs, lr, sub_key_index, attack_size, rank_step, unmask, network_name):
    ranks_x = []
    ranks_y = []

    for run in runs:
        model_path = '/media/rico/Data/TU/thesis/runs2/subkey_{}/{}_SF{}_E{}_BZ{}_LR{}/train{}/model_r{}_{}.pt'.format(
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

        if "Dense" in network_name:
            model = DenseSpreadNet.DenseSpreadNet.load_model(model_path)
        elif "MLP" in network_name:
            model = DenseNet.load_model(model_path)
        # elif "SpreadNet" in network_name:
        #     model = SpreadNetIn.load_spread(model_path)
        elif "SpreadNet" in network_name:
            model = SpreadNet.load_spread(model_path)
        else:
            raise Exception("Unkown model")
        print("Using {}".format(model))
        model.to(device)

        (_, _), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(trace_file, load_metadata=True)

        x, y = test(x_attack, y_attack, metadata_attack,
                    network=model,
                    sub_key_index=sub_key_index,
                    use_hw=use_hw,
                    attack_size=attack_size,
                    rank_step=rank_step,
                    unmask=unmask,
                    only_accuracy=False)

        if isinstance(model, SpreadNetIn):
            v = model.intermediate_values
            order = [int((x % spread_factor) * 100 + math.floor(x / spread_factor)) for x in range(spread_factor * 100)]
            inter = []
            for x in range(len(v[0])):
                inter.append([v[0][x][j] for j in order])

            std = np.std(inter, axis=0)
            div_by = 1.0 / attack_size * 10
            print("divby: {}".format(div_by))
            res = np.where(std < div_by, 1, 0)

            mean_res = np.mean(inter, axis=0)
            mean_res2 = np.where(mean_res < div_by, 1, 0)
            print('Sum  std results {}'.format(np.sum(res)))
            print('Sum mean results {}'.format(np.sum(mean_res2)))

            total_same = 0
            for j in range(len(mean_res2)):
                if mean_res2[j] == 1 and res[j] == 1:
                    total_same += 1
            print('Total same: {}'.format(total_same))

            plt.title('Performance of networks')
            plt.xlabel('#neuron')
            plt.ylabel('std')
            xcoords = [j * spread_factor for j in range(100)]
            for xc in xcoords:
                plt.axvline(x=xc, color='green')
            plt.grid(True)
            plt.plot(std, label='std')
            plt.figure()

            plt.title('Performance of networks')
            plt.xlabel('#neuron')
            plt.ylabel('mean')
            for xc in xcoords:
                plt.axvline(x=xc, color='red')
            plt.grid(True)
            plt.plot(mean_res, label='mean')
            plt.legend()
            plt.show()

        ranks_x.append(x)
        ranks_y.append(y)

        # accuracy()
        data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
        print('x_test size: {}'.format(data.cpu().size()))
        predictions = F.softmax(model(data).to(device), dim=-1).to(device)
    return ranks_x, ranks_y


ranks_x = []
ranks_y = []
rank_mean_y = []
for network_name in network_names:
    x, y = get_ranks(use_hw, runs, train_size, epochs, lr, sub_key_index
                                 , attack_size, rank_step, unmask, network_name)
    mean_y = np.mean(y, axis=0)
    ranks_x.append(x)
    ranks_y.append(y)
    rank_mean_y.append(mean_y)


for i in range(len(rank_mean_y)):
    plt.title('Performance of {}'.format(network_names[i]))
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)

    # Plot the results
    for x, y in zip(ranks_x[i], ranks_y[i]):
        plt.plot(x, y)
    plt.figure()


plt.title('Performance of networks')
plt.xlabel('Number of traces')
plt.ylabel('Mean rank')
plt.grid(True)
for i in range(len(rank_mean_y)):
    plt.plot(ranks_x[i][0], rank_mean_y[i], label=network_names[i])
    plt.legend()

    # plt.figure()

plt.show()
