from decimal import Decimal

import torch

from models import DenseSpreadNet
from models.DenseNet import DenseNet
from models.SpreadNet import SpreadNet
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
spread_factor = 6
runs = [x for x in range(5)]
train_size = 1000
epochs = 80
batch_size = 100
lr = 0.00001
sub_key_index = 2
attack_size = 2000
rank_step = 1
type_network = 'HW' if use_hw else 'ID'
unmask = False if sub_key_index < 2 else True
# network_name = 'SpreadNet'
# network_name = 'DenseSpreadNet'
# network_name = "MLPBEST"

network_names = ['SpreadNet', 'MLPBEST', 'DenseSpreadNet']
#####################################################################################

trace_file = '{}/data/ASCAD_{}.h5'.format(path, sub_key_index)
device = torch.device("cuda")


def get_ranks(use_hw, runs, train_size,
              epochs, lr, sub_key_index, attack_size, rank_step, unmask, network_name):
    ranks_x = []
    ranks_y = []

    for run in runs:
        model_path = '/media/rico/Data/TU/thesis/runs/subkey_{}/{}_SF{}_E{}_BZ{}_LR{}/train{}/model_r{}_{}.pt'.format(
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
        else:
            model = SpreadNet.load_spread(model_path)
        print("Using {}".format(model))
        model.to(device)

        (_, _), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(trace_file, load_metadata=True)

        x, y = test(x_attack, y_attack, metadata_attack,
                    network=model,
                    sub_key_index=sub_key_index,
                    use_hw=use_hw,
                    attack_size=attack_size,
                    rank_step=rank_step,
                    unmask=unmask)

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
    for x, y in zip(ranks_x[i], ranks_y[i]):
        # Plot the results

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
