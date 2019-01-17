import torch

from models import DenseSpreadNet
from models.SpreadNet import SpreadNet
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ascad import load_ascad
from test import test

path = '/media/rico/Data/TU/thesis'

#####################################################################################
# Parameters
use_hw = True
n_classes = 9 if use_hw else 256
spread_factor = 6
runs = [x for x in range(5)]
train_size = 50000
epochs = 80
batch_size = 100
lr = 0.00001
sub_key_index = 0
attack_size = 10000
rank_step = 5
type_network = 'HW' if use_hw else 'ID'
network_name = 'DenseSpreadNet'
# network_name = 'SpreadNet'

#####################################################################################

trace_file = '{}/data/ASCAD_{}.h5'.format(path, sub_key_index)
device = torch.device("cuda")


ranks_x = []
ranks_y = []

for run in runs:
    model_path = '/media/rico/Data/TU/thesis/runs/subkey_{}/{}_SF{}_E{}_BZ{}_LR1E-5/train{}/model_r{}_{}.pt'.format(
        sub_key_index,
        type_network,
        spread_factor,
        epochs,
        batch_size,
        train_size,
        run,
        network_name
    )
    print('path={}'.format(model_path))

    model = DenseSpreadNet.DenseSpreadNet.load_model(model_path)
    # model = SpreadNet.load_spread(model_path)
    model.to(device)

    (x_profiling, y_profiling), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(trace_file,
                                                                                                         load_metadata=True)

    x, y = test(x_attack, y_attack, metadata_attack,
                network=model,
                sub_key_index=sub_key_index,
                use_hw=use_hw,
                attack_size=attack_size,
                rank_step=rank_step)

    ranks_x.append(x)
    ranks_y.append(y)

    # accuracy()
    data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
    print('x_test size: {}'.format(data.cpu().size()))
    predictions = F.softmax(model(data).to(device), dim=-1).to(device)
    d = predictions[0].detach().cpu().numpy()
    print(d)


plt.title('Performance of {}'.format(network_name))
plt.xlabel('number of traces')
plt.ylabel('rank')
plt.grid(True)
for x, y in zip(ranks_x, ranks_y):
    # Plot the results

    plt.plot(x, y)
plt.figure()

rank_avg_y = np.mean(ranks_y, axis=0)

plt.title('Performance of {}'.format(network_name))
plt.xlabel('number of traces')
plt.ylabel('rank')
plt.grid(True)
plt.plot(ranks_x[0], rank_avg_y, label='mean')
plt.legend()
plt.show()
plt.figure()
