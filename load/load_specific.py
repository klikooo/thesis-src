import itertools
from decimal import Decimal

import util
import numpy as np

import matplotlib.pyplot as plt

from util_classes import get_save_name

path = '/media/rico/Data/TU/thesis'

#####################################################################################
# Parameters
use_hw = False
n_classes = 9 if use_hw else 256
spread_factor = 1
runs = [x for x in range(1)]
train_size = 40000
epochs = 75
batch_size = 100
lr = 0.0001
sub_key_index = 2
rank_step = 1

unmask = True
kernel_sizes = [3, 10, 25, 50, 100]
num_layers = [2]
channel_sizes = [128]
l2_penalty = 0.00
max_pool = 64
init_weights = "kaiming"

# network_names = ['SpreadV2', 'SpreadNet', 'DenseSpreadNet', 'MLPBEST']
network_settings = {
    'SmallCNN': 1,
    # 'KernelBigVGGMDK': {}
}
data_set = util.DataSet.RANDOM_DELAY_NORMALIZED
plt_titles = ['$Spread_{PH}$', '$Dense_{RT}$', '$MLP_{best}$', '', '', '', '']
only_accuracy = False
desync = 0
show_losses = True
show_acc = False
show_losses_all = False
show_only_mean = True
experiment = False

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
                   "max_pool": max_pool}
        network_settings[k].append(setting)

#####################################
# UPDATE SETTINGS FOR DESIRED MODEL #
#####################################
# network_settings['NumLayersVGG3'][0].update({
#     "kernel_sizes": [17, 30],
#     "num_layers": [4]})
# network_settings['NumLayersVGG3'][1].update({
#     "kernel_sizes": [20],
#     "num_layers": [5]})

#####################################################################################


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

        all_loss_acc.append(loss_acc)

    for setting in network_setting:
        model_params.update({"max_pool": setting['max_pool']})
        util.loop_at_least_once(setting['kernel_sizes'], lambda_kernel, lambda: (
            util.loop_at_least_once(setting['channel_sizes'], lambda_channel, lambda: (
                util.loop_at_least_once_with_arg(setting['num_layers'], lambda_layers, retrieve_ge,
                                                 setting)
            ))
        ))

###############################################
# Plot the runs of the same model in one plot #
###############################################
line_marker = itertools.cycle(('+', '.', 'o', '*'))
for i in range(len(rank_mean_y)):
    plt.title('Performance of {}'.format(name_models[i]), fontsize=20)
    plt.xlabel('Number of traces', fontsize=16)
    plt.ylabel('Guessing Entropy', fontsize=16)
    plt.grid(True)
    axes = plt.gca()
    axes.set_ylim([0, 256])

    # Plot the results
    for x, y in zip(ranks_x[i], ranks_y[i]):
        plt.plot(x, y)
    figure = plt.gcf()
    plt.figure()
    figure.savefig('/home/rico/Pictures/{}.png'.format(name_models[i]), dpi=100)

###############################################
# Plot the mean of the runs of a single model #
###############################################
plt.xlabel('Number of traces', fontsize=16)
plt.ylabel('Guessing Entropy', fontsize=16)
plt.grid(True)
axes = plt.gca()
axes.set_ylim([0, 256])
for i in range(len(rank_mean_y)):
    plt.plot(ranks_x[i][0], rank_mean_y[i], label=name_models[i], marker=next(line_marker))
    plt.legend()

    # plt.figure()
figure = plt.gcf()
figure.savefig('/home/rico/Pictures/{}.png'.format('mean'), dpi=100)

################################
# Show loss and accuracy plots #
################################
if show_losses or show_acc:
    mean_mv = []
    mean_lv = []

    ############
    # ACCURACY #
    ############
    for i in range(len(rank_mean_y)):
        (loss_vali, loss_train, acc_train, acc_vali) = all_loss_acc[i]
        if not show_only_mean:
            plt.figure()
            for r in range(len(loss_vali)):
                plt.title('Accuracy during training {}'.format(name_models[i]))
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.grid(True)
                # Plot the accuracy
                # for x, y in zip(ranks_x[i], ranks_y[i]):
                # pdb.set_trace()
                plt.plot([x for x in range(len(acc_train[r]))], acc_train[r] * 100, label="Train", color='orange')
                plt.plot([x for x in range(len(acc_train[r]))], acc_vali[r] * 100, label="Vali", color='green')
                plt.legend()
        mt = np.mean(acc_train, axis=0) * 100
        mv = np.mean(acc_vali, axis=0) * 100
        if not show_only_mean:
            plt.plot(mt, color='blue')
            plt.plot(mv, color='red')
        mean_mv.append(mv)

    ########
    # LOSS #
    ########
    for i in range(len(rank_mean_y)):
        (loss_train, loss_vali, acc_train, acc_vali) = all_loss_acc[i]
        if not show_only_mean:
            plt.figure()
            for r in range(len(loss_vali)):
                plt.title('Loss during training {}'.format(name_models[i]))
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                # Plot the accuracy
                # for x, y in zip(ranks_x[i], ranks_y[i]):
                plt.plot([x for x in range(len(loss_train[r]))], loss_train[r], label="Train", color='orange')
                plt.plot([x for x in range(len(loss_train[r]))], loss_vali[r], label="Vali", color='green')
                plt.legend()

        lt = np.mean(loss_train, axis=0)
        lv = np.mean(loss_vali, axis=0)
        if not show_only_mean:
            plt.plot(lt, color='blue', label='Train')
            plt.plot(lv, color='red', label='Validation')
        mean_lv.append(lv)

    ##############
    # SHOW MEANS #
    ##############
    plt.figure()
    for i in range(len(mean_lv)):
        plt.plot(mean_lv[i], label="Loss {}".format(name_models[i]))
    plt.grid(True)
    plt.title("Mean loss validation")
    plt.legend()

    plt.figure()
    for i in range(len(mean_mv)):
        plt.plot(mean_mv[i], label="Accuracy {}".format(name_models[i]))
    plt.grid(True)
    plt.title("Mean accuracy validation")
    plt.legend()

plt.show()
