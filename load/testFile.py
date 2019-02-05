import math
from decimal import Decimal

import torch

from models import DenseSpreadNet
from models.CosNet import CosNet
from models.DenseNet import DenseNet
from models.SpreadNet import SpreadNet
from models.SpreadNetIn import SpreadNetIn
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.SpreadV2 import SpreadV2
from util import load_ascad, shuffle_permutation, load_csv, load_aes_hd, load_dpav4, SBOX_INV
from test import test, test_with_key_guess

path = '/media/rico/Data/TU/thesis'

#####################################################################################
# Parameters
use_hw = False
n_classes = 9 if use_hw else 256
spread_factor = 6
runs = [x for x in range(5)]
train_size = 10000
epochs = 80
batch_size = 100
lr = 0.001
sub_key_index = 2
attack_size = 10000
rank_step = 1
type_network = 'HW' if use_hw else 'ID'
unmask = False if sub_key_index < 2 else True

# network_names = ['SpreadV2', 'SpreadNet', 'DenseSpreadNet', 'MLPBEST']
network_names = ['SpreadV2', 'DenseSpreadNet']
plt_titles = ['$Spread_{PH}$', '$Dense_{RT}$', '$MLP_{best}$']
if len(plt_titles) != len(network_names):
    plt_titles = network_names
# network_names = ['CosNet']
# network_names = ['MLPBEST']
only_accuracy = False

#####################################################################################

device = torch.device("cuda")

x_attack, y_attack = load_dpav4({'use_hw': use_hw,
                                  'traces_path': '/media/rico/Data/TU/thesis/data'})
key_guesses = np.transpose(
    load_csv('/media/rico/Data/TU/thesis/data/DPAv4/Value/key_guesses_ALL.csv', delimiter=' ', dtype=np.int))


def get_plaintexts(y, key=249, len2=1000):
    plain = np.zeros(len2)
    for i in range(len2):
        plain[i] = SBOX_INV[y[i]] ^ key
    return plain


plains = get_plaintexts(y_attack)
print(key_guesses[0][y_attack[0]])
print(key_guesses[1][y_attack[1]])
print(key_guesses[2][y_attack[2]])




# x_attack = x_attack[train_size:train_size + attack_size]
# y_attack = y_attack[train_size:train_size + attack_size]
# key_guesses = key_guesses[train_size:train_size + attack_size]
