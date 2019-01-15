import torch

from models.SpreadNet import SpreadNet
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ascad import load_ascad
from test import test

path = '/media/rico/Data/TU/thesis'
file = '{}/data/ASCAD_0.h5'.format(path)

#####################################################################################
# Parameters
use_hw = True
n_classes = 9 if use_hw else 256
spread_factor = 6
runs = 5
train_size = 2000
epochs = 1400
batch_size = 100
lr = 0.00001
sub_key_index = 0
attack_size = 1000
rank_step = 10
type_network = 'HW' if use_hw else 'ID'
#####################################################################################

model_path = '/media/rico/Data/TU/thesis/runs/subkey_{}/{}_SF{}_E{}_BZ{}_LR1E-5/train{}/model_r0_Spread network.pt'.format(
    sub_key_index,
    type_network,
    spread_factor,
    epochs,
    batch_size,
    train_size
)


device = torch.device("cuda")

# model = SpreadNet(spread_factor, input_shape=700, out_shape=n_classes)
# model.load_state_dict(torch.load(PATH))
# print(model.state_dict())
model = SpreadNet.load_spread(model_path)
model.to(device)


(x_profiling, y_profiling), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(file,
                                                                                                     load_metadata=True)

x, y = test(x_attack, y_attack, metadata_attack,
            network=model,
            sub_key_index=sub_key_index,
            use_hw=use_hw,
            attack_size=attack_size,
            rank_step=rank_step)


# accuracy()
data = torch.from_numpy(x_attack.astype(np.float32)).to(device)
print('x_test size: {}'.format(data.cpu().size()))
predictions = F.softmax(model(data).to(device), dim=-1).to(device)
d = predictions[0].detach().cpu().numpy()
print(d)


# Plot the results
plt.title('Performance of {}'.format('a'))
plt.xlabel('number of traces')
plt.ylabel('rank')
plt.grid(True)
plt.plot(x, y)
plt.show()
