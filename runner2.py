from models.DenseNet import DenseNet
from models.SpreadNet import SpreadNet
from models.DenseSpreadNet import DenseSpreadNet
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import torch

from models.TestNet import TestNet
from ascad import load_ascad
from test import test
from train import train

path = '/media/rico/Data/TU/thesis'
ranks_x = []
ranks_y = []

#####################################################################################
# Parameters
use_hw = True
spread_factor = 6
runs = 5
train_size = 1000
epochs = 1400
batch_size = 100
lr = 0.00001
sub_key_index = 0
attack_size = 2000
rank_step = 10
#####################################################################################
# Select the number of classes to use depending on hw
n_classes = 9 if use_hw else 256
type_network = ''
file = '{}/data/ASCAD_{}.h5'.format(path, sub_key_index)

network = None
title = 'Torch {} \nSpread factor {}\nTrain size {}, batch {}, lr {}, epochs {}, Type {}'.format(
    type_network,
    spread_factor,
    train_size,
    batch_size,
    lr,
    epochs,
    'HW' if use_hw else 'ID')

# Save the ranks to a file
dir_name = 'subkey_{}/{}_SF{}_E{}_BZ{}_LR1E-5/train{}'.format(
    sub_key_index,
    'HW' if use_hw else 'ID',
    spread_factor,
    epochs,
    batch_size,
    train_size
)

# Load data
(x_profiling, y_profiling), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(file,
                                                                                                     load_metadata
                                                                                                     =True)
for i in range(runs):
    # Choose which network to use
    # network = SpreadNet(spread_factor=spread_factor, input_shape=700, out_shape=n_classes)
    network = DenseSpreadNet(spread_factor=spread_factor, input_shape=700, out_shape=n_classes)
    # network = TestNet(pr_shape=460, sbox_shape=500, n_classes=n_classes)
    # network = DenseNet(input_shape=700, n_classes=n_classes)

    network = train(x_profiling, y_profiling,
                    train_size=train_size,
                    network=network,
                    epochs=epochs,
                    batch_size=batch_size,
                    use_hw=use_hw,
                    lr=lr
                    )
    if isinstance(network, SpreadNet):
        network.training = False

    x, y = test(x_attack, y_attack, metadata_attack,
                network=network,
                sub_key_index=sub_key_index,
                use_hw=use_hw,
                attack_size=attack_size,
                rank_step=rank_step)
    ranks_x.append(np.array(x))
    ranks_y.append(np.array(y))

    type_network = network.name()

    model_save_file = '{}/runs/{}/model_r{}_{}.pt'.format(path, dir_name, i, type_network)
    os.makedirs(os.path.dirname(model_save_file), exist_ok=True)

    if isinstance(network, SpreadNet):
        network.save(model_save_file)
    elif isinstance(network, DenseSpreadNet):
        network.save(model_save_file)
    else:
        torch.save(network.state_dict(), model_save_file)

x_save_file = '{}/runs/{}/x_{}.r'.format(path, dir_name, type_network)
y_save_file = '{}/runs/{}/y_{}.r'.format(path, dir_name, type_network)

os.makedirs(os.path.dirname(x_save_file), exist_ok=True)
with open(x_save_file, 'wb') as f:
    pickle.dump(ranks_x, f)

with open(y_save_file, 'wb') as f:
    pickle.dump(ranks_y, f)

# Plot the results
plt.title('Performance of {}'.format(title))
plt.xlabel('number of traces')
plt.ylabel('rank')
plt.grid(True)
for i in range(runs):
    plt.plot(ranks_x[i], ranks_y[i])
plt.figure()

# Show a figure with the mean of the runs
rank_avg_y = np.mean(ranks_y, axis=0)

plt.title('Performance of {}'.format(title))
plt.xlabel('number of traces')
plt.ylabel('rank')
plt.grid(True)
plt.plot(ranks_x[0], rank_avg_y, label='mean')
plt.legend()
plt.show()
plt.figure()
