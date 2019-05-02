from models.Spread.SpreadNet import train_test, SpreadNet
from models.Spread.DenseSpreadNet import DenseSpreadNet
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import torch

path = '/media/rico/Data/TU/thesis'
file = '{}/data/ASCAD_0.h5'.format(path)
ranks_x = []
ranks_y = []

# Parameters
use_hw = False
spread_factor = 6
runs = 3
train_size = 10
epochs = 10
batch_size = 1
lr = 0.00001
sub_key_index = 0
attack_size = 100
rank_step=2
# Select the number of classes to use depending on hw
n_classes = 9 if use_hw else 256
type_network = ''


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
dir_name = '{}_SF{}_E{}_BZ{}_LR1E-5/train{}'.format(
    'HW' if use_hw else 'ID',
    spread_factor,
    epochs,
    batch_size,
    train_size
)

for i in range(runs):
    # Choose which network to use
    network = SpreadNet(spread_factor=spread_factor, input_shape=700, out_shape=n_classes)
    # network = DenseSpreadNet(spread_factor=spread_factor, input_shape=700, out_shape=n_classes)

    x, y = train_test(file, sub_key_index=sub_key_index, network=network, train_size=train_size, epochs=epochs,
                      batch_size=batch_size, lr=lr,
                      use_hw=use_hw,
                      attack_size=attack_size,
                      rank_step=rank_step)
    ranks_x.append(np.array(x))
    ranks_y.append(np.array(y))
    if isinstance(network, DenseSpreadNet):
        type_network = 'Dense-spread hidden network'
    elif isinstance(network, SpreadNet):
        type_network = 'Spread Non Relu network'
    else:
        type_network = '?? network'

    model_save_file = '{}/runs/{}/model_r{}_{}.pt'.format(path, dir_name, i, type_network)
    print(model_save_file)
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
