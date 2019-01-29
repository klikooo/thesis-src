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
spread_factor = 9
runs = [x for x in range(10)]
train_size = 2000
epochs = 80
batch_size = 1000
lr = 0.0001
sub_key_index = 2
attack_size = 1000
rank_step = 1
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
only_accuracy = False


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
                    only_accuracy=only_accuracy)

        if isinstance(model, SpreadNetIn):
            # Get the intermediate values right after the first fully connected layer
            z = np.transpose(model.intermediate_values2[0])

            # Calculate the mse for the maximum and minumum from these traces and the learned min and max
            min_z = np.min(z, axis=1)
            max_z = np.max(z, axis=1)
            msq_min = np.mean(np.square(min_z - model.tensor_min), axis=None)
            msq_max = np.mean(np.square(max_z - model.tensor_max), axis=None)
            print('msq min: {}'.format(msq_min))
            print('msq max: {}'.format(msq_max))

            # Plot the distribution of each neuron right after the first fully connected layer
            # for k in range(100):
            #     plt.grid(True)
            #     plt.axvline(x=model.tensor_min[k], color='green')
            #     plt.axvline(x=model.tensor_max[k], color='green')
            #     plt.hist(z[:][k], bins=40)
            #
            #     plt.show()
            # exit()

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
            mean_res2 = np.where(mean_res < threshold, 1, 0)
            print('Sum  std results {}'.format(np.sum(res)))
            print('Sum mean results {}'.format(np.sum(mean_res2)))

            # Check which neurons have a std and mean where it is smaller than threshold
            total_same = 0
            for j in range(len(mean_res2)):
                if mean_res2[j] == 1 and res[j] == 1:
                    total_same += 1
            print('Total same: {}'.format(total_same))

            # Plot the standard deviations
            plt.title('Performance of networks')
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
