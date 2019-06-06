import itertools
from decimal import Decimal

import util
import numpy as np

import matplotlib.pyplot as plt

from util_classes import get_save_name

import matplotlib
matplotlib.rcParams.update({'font.size': 18})

path = '/media/rico/Data/TU/thesis'

#####################################################################################
# Parameters
use_hw = True
spread_factor = 6
runs = [x for x in range(1)]
train_size = 1000
epochs = 80
batch_size = 100
lr = 0.0001
sub_key_index = 2
rank_step = 1

unmask = True  # False if sub_kezy_index < 2 else True
kernel_sizes = [0]
num_layers = [0]
channel_sizes = [0]
l2_penalty = 0
init_weights = ""


data_set = util.DataSet.ASCAD
plt_titles = ['$Spread_{PH}$', '$MLP_{RT}$', '$MLP_{best}$', '', '', '', '']
only_accuracy = False
desync = 0
show_losses = True
show_losses_all = False
show_only_mean = True
show_ge = False
experiment = False
show_acc = False
show_loss = False
colors = ["aqua", "black", "brown", "darkblue", "darkgreen",
          "fuchsia", "goldenrod", "green", "grey", "indigo", "lavender"]
plot_markers = [" ", "*", ".", "o", "+", "D"]


network_1 = "SpreadNet"
network_2 = "DenseNet"
network_3 = "DenseSpreadNet"
network_settings = {
    network_1: 1,
    network_2: 1,
    network_3: 1,
}
n_classes = 9 if use_hw else 256
###########################
# SETTINGS FOR EACH MODEL #
###########################
for k, v in network_settings.items():
    network_settings[k] = []
    for num_models in range(v):
        setting = {"experiment": '3' if not experiment else '',
                   "data_set": data_set,
                   "subkey_index": sub_key_index,
                   "unmask": unmask,
                   "desync": desync,
                   "use_hw": use_hw,
                   "spread_factor": spread_factor,
                   "epochs": epochs,
                   "batch_size": batch_size,
                   "lr": '%.2E' % Decimal(lr),
                   "l2_penalty": l2_penalty,
                   "train_size": train_size,
                   "kernel_sizes": kernel_sizes,
                   "num_layers": num_layers,
                   "channel_sizes": channel_sizes,
                   "network_name": k,
                   "init_weights": init_weights,
                   "title": "",
                   "plot_colors": colors,
                   "ge_x": [],
                   "ge_y": [],
                   "ta": [],
                   "va": [],
                   "tl": [],
                   "vl": [],
                   "line_title": []
                   }
        network_settings[k].append(setting)

#####################################
# UPDATE SETTINGS FOR DESIRED MODEL #
#####################################
# network_settings[network_1][0].update({
#     "kernel_sizes": [100, 50, 26, 21, 17, 15],
#     "num_layers": [1, 2, 3, 4, 5, 6],
#     "l2_penalty": 0.05,
#     "title": " l2 0.05 different kernels",
#     "plot_marker": " ",
# })
network_settings[network_1][0].update({
    "plot_marker": " ",
    "plot_colors": ['g'],
    "title": "asd",
    "line_title2": "$Spread_{PH}$"
})
network_settings[network_2][0].update({
    "plot_marker": "<",
    "plot_colors": ['r'],
    "title": "",
    "line_title2": "$MLP_{best}$"
})
network_settings[network_3][0].update({
    "plot_marker": "o",
    "plot_colors": ['b'],
    "title": " ",
    "line_title2": "$MLP_{RT}$",
})
#####################################################################################


n_settings = []


# Function to load the GE of a single model
def get_ge(net_name, model_parameters, load_parameters):
    args = util.EmptySpace()
    for key, value in load_parameters.items():
        setattr(args, key, value)
    folder = "/media/rico/Data/TU/thesis/runs{}/{}".format(args.experiment, util.generate_folder_name(args))

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


########################################
# Load the GE results  of the networks #
########################################
ranks_x = []
ranks_y = []
rank_mean_y = []
name_models = []
model_params = {}
all_loss_acc = []  # ([], [], [], [])
plot_colors = []
for network_name, network_setting in network_settings.items():
    def lambda_kernel(x): model_params.update({"kernel_size": x})


    def lambda_channel(x): model_params.update({"channel_size": x})


    def lambda_layers(x): model_params.update({"num_layers": x})


    def retrieve_ge(net_setting):
        print(model_params)
        ge_x, ge_y, loss_acc = get_ge(network_name, model_params, net_setting)
        mean_y = np.mean(ge_y, axis=0)
        ranks_x.append(ge_x)
        ranks_y.append(ge_y)
        rank_mean_y.append(mean_y)
        name_models.append(get_save_name(network_name, model_params))
        n_settings.append(net_setting)

        (lvl, ltl, lta, lva) = loss_acc
        # loss_vali, loss_train, acc_train, acc_vali

        net_setting['ge_x'].append(ge_x[0])
        net_setting['ge_y'].append(mean_y)

        net_setting['ta'].append(np.mean(lta, axis=0))
        net_setting['va'].append(np.mean(lva, axis=0))
        net_setting['tl'].append(np.mean(ltl, axis=0))
        net_setting['vl'].append(np.mean(lvl, axis=0))
        net_setting['line_title'].append(get_save_name(network_name, model_params))

        all_loss_acc.append(loss_acc)

    for setting in network_setting:
        print(setting)
        # exit()
        for cs in setting['channel_sizes']:
            model_params.update({"channel_size": cs})
            for i in range(len(setting['num_layers'])):
                model_params.update({"kernel_size": setting['kernel_sizes'][i]})
                model_params.update({"num_layers": setting['num_layers'][i]})
                plot_colors.append(setting['plot_colors'][i])
                retrieve_ge(setting)


###############################################
# Plot the runs of the same model in one plot #
###############################################
line_marker = itertools.cycle(('+', '.', 'o', '*'))

for model_name, model_settings in network_settings.items():
    for model_setting in model_settings:
        # Plot GE
        plt.figure()
        plt.xlabel('Number of traces', fontsize=16)
        plt.ylabel('Guessing Entropy', fontsize=16)
        plt.grid(True)
        axes = plt.gca()
        axes.set_ylim([0, 256])
        plt.title("{} - {}".format(model_name, model_setting['title']))

        # print(model_setting)

        for i in range(len(model_setting['ge_x'])):
            plt.plot(model_setting['ge_x'][i], model_setting['ge_y'][i],
                     label="{} - {}".format(model_name, model_setting['line_title'][i]),
                     color=model_setting['plot_colors'][i])
        plt.legend()

        # Plot accuracy if asked for
        if show_acc:
            plt.figure()
            plt.title("Accuracy during training {} - {}".format(model_name, model_setting['title']))
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            for i in range(len(model_setting['ge_x'])):
                plt.plot(model_setting['ta'][i] * 100, label="Train {}".format(model_setting['line_title'][i]),
                         color='orange', marker=plot_markers[i])
                plt.plot(model_setting['va'][i] * 100, label="Validation {}".format(model_setting['line_title'][i]),
                         color='green', marker=plot_markers[i])
            plt.legend()

        if show_loss:
            plt.figure()
            plt.title("Loss during training {} - {}".format(model_name, model_setting['title']))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            for i in range(len(model_setting['ge_x'])):
                plt.plot(model_setting['tl'][i], label="Train {}".format(model_setting['line_title'][i]),
                         color='orange', marker=plot_markers[i])
                plt.plot(model_setting['vl'][i], label="Validation {}".format(model_setting['line_title'][i]),
                         color='green', marker=plot_markers[i])
            plt.legend()

    # Plot all GE in same plot

plt.figure()
plt.xlabel('Number of traces', fontsize=16)
plt.ylabel('Guessing Entropy', fontsize=16)
plt.grid(True)
axes = plt.gca()
axes.set_ylim([0, 95])
axes.set_xlim([-1, 65])
# plt.title("{} - {}".format(model_name, model_setting['title']))
for model_name, model_settings in network_settings.items():

    for model_setting in model_settings:
        for i in range(len(model_setting['ge_x'])):
            plt.plot(model_setting['ge_x'][i], model_setting['ge_y'][i],
                     label=f"{model_setting['line_title2']}",
                     color=model_setting['plot_colors'][i])
plt.legend()
# print(dir(mng))
# exit()
figure = plt.gcf()
i = 1
figure.set_size_inches(16*i, 9*i)
figure.savefig('/media/rico/Data/TU/thesis/report/img/spread/repro/same_settings.png')

plt.show()

