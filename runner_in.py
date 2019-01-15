from models.SpreadNet import train_test, SpreadNet
from models.DenseSpreadNet import DenseSpreadNet
from models.SpreadNetIn import SpreadNetIn
import matplotlib.pyplot as plt
import numpy as np

path = '/media/rico/Data/TU/thesis'
file = '{}/data/ASCAD_0.h5'.format(path)

# Parameters
use_hw = True
spread_factor = 6
runs = 1
train_size = 1000
epochs = 1400
batch_size = 1000
lr = 0.00001
sub_key_index = 0

# Select the number of classes to use depending on hw
n_classes = 9 if use_hw else 256
type_network = ''
intermediate_values = []

for i in range(runs):
    # Choose which network to use
    network = SpreadNetIn(spread_factor=spread_factor, input_shape=700, out_shape=n_classes)
    # network = DenseSpreadNet(spread_factor=spread_factor, input_shape=700, out_shape=n_classes)

    x, y = train_test(file, sub_key_index=sub_key_index, network=network, train_size=train_size, epochs=epochs,
                      batch_size=batch_size, lr=lr,
                      use_hw=use_hw,
                      attack_size=10)
    intermediate_values = network.intermediate_values
    # ranks_x.append(np.array(x))
    # ranks_y.append(np.array(y))
    if isinstance(network, DenseSpreadNet):
        type_network = 'Dense-spread network'
    elif isinstance(network, SpreadNet):
        type_network = 'Spread network'
    else:
        type_network = '?? network'

title = 'Torch {} \nSpread factor {}\nTrain size {}, batch {}, lr {}, epochs {}, Type {}'.format(
    type_network,
    spread_factor,
    train_size,
    batch_size,
    lr,
    epochs,
    'HW' if use_hw else 'ID')


# for i in range(100):
#     print('VAL {}, {}, {}, {}, {}, {}'.format(intermediate_values[0][0][i], intermediate_values[0][0][i + 100],
#                                               intermediate_values[0][0][i + 200], intermediate_values[0][0][i + 300],
#                                               intermediate_values[0][0][i + 400], intermediate_values[0][0][i + 500]))

# epochs = list(range(0, 10)) + list(range(690, 700))
l_epochs = range(0, epochs+1, 100)
print(l_epochs)
x_as = range(len(intermediate_values[0][0]))
z = []
zeroes = []
std = []
for i in range(len(l_epochs)):
    zero = np.count_nonzero(intermediate_values[l_epochs[i]], axis=1)
    std.append(np.std(zero))
    zeroes.append(np.mean(zero))
    z.append(np.mean(intermediate_values[l_epochs[i]], axis=0))


# First 100 values
f100 = z[len(l_epochs)-1][0:100]
print('F100: {} {}'.format(np.count_nonzero(f100 == 0), np.count_nonzero(f100 == 1)))
print(zeroes[0])
print(std[0])
plt.title('Intermediate values')
plt.grid(True)
for i in range(len(z)):
    plt.plot(x_as, z[i], label='Epoch {}'.format(l_epochs[i]))
plt.legend()
plt.show()


# Show a figure with the mean of the runs
# rank_avg_y = np.mean(ranks_y, axis=0)
