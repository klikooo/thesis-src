import itertools
from decimal import Decimal

import util
import numpy as np

import matplotlib.pyplot as plt

from util_classes import get_save_name
import os

import matplotlib
from matplotlib.lines import Line2D

# matplotlib.rcParams.update({'font.size': 18})


def plot_rd(noise_level, x_limits, y_limits, show=True, file_extension=""):
    #####################################################################################
    # Parameters
    use_hw = False
    spread_factor = 1
    runs = [x for x in range(5)]
    train_size = 40000
    epochs = 75
    batch_size = 100
    lr = 0.0001
    sub_key_index = 2
    l2_penalty = 0.0

    unmask = True  # False if sub_kezy_index < 2 else True
    kernel_sizes = []
    num_layers = []
    channel_sizes = [32]
    init_weights = "kaiming"

    network_1 = "VGGNumLayers2"
    network_settings = {
        network_1: 1,
    }
    data_set = util.DataSet.RANDOM_DELAY_NORMALIZED
    desync = 0
    load_loss_acc = True
    show_losses = False
    show_acc = False
    show_only_mean = False
    show_ge = False
    experiment = False
    show_loss = False
    show_per_layer = True
    colors = ["aqua", "black", "brown", "darkblue", "darkgreen",
              "fuchsia", "goldenrod", "grey", "indigo", "lavender"]
    plot_markers = [" ", "*", ".", "o", "+", "8", "s", "p", "P", "h", "H"]
    # "8"	m11	octagon
    # "s"	m12	square
    # "p"	m13	pentagon
    # "P"	m23	plus (filled)
    # "*"	m14	star
    # "h"	m15	hexagon1
    # "H"	m16	hexagon2

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
    kernels = [15] * 5
    network_settings[network_1][0].update({
        "kernel_sizes": kernels,
        "num_layers": list(range(1, len(kernels) + 1)),
        "l2_penalty": l2_penalty,
        "title": " 15 kernel, l2 {}".format(l2_penalty),
        "plot_marker": " ",
    })
    #####################################################################################
    noise_string = f'_noise{noise_level}' if noise_level > 0.0 else ''
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
            ge_path = '{}{}.exp__'.format(filename, noise_string)
            if not os.path.exists(ge_path):
                ge_path = f"{filename}{noise_string}.exp"

            y_r = util.load_csv(ge_path, delimiter=' ', dtype=np.float)
            x_r = range(len(y_r))
            ge_x.append(x_r)
            ge_y.append(y_r)

            if load_loss_acc:
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
        def retrieve_ge(net_setting):
            print(model_params)
            ge_x, ge_y, loss_acc = get_ge(network_name, model_params, net_setting)
            mean_y = np.mean(ge_y, axis=0)
            ranks_x.append(ge_x)
            ranks_y.append(ge_y)
            rank_mean_y.append(mean_y)
            name_models.append(get_save_name(network_name, model_params))
            n_settings.append(net_setting)

            (lta, lva, ltl, lvl) = loss_acc

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
            model_params.update({"max_pool": 4})
            for cs in setting['channel_sizes']:
                model_params.update({"channel_size": cs})
                for i in range(len(setting['num_layers'])):
                    model_params.update({"kernel_size": setting['kernel_sizes'][i]})
                    model_params.update({"num_layers": setting['num_layers'][i]})
                    plot_colors.append(setting['plot_colors'][i])
                    retrieve_ge(setting)

    ###############################################
    # Plot the mean of the runs of a single model #
    ###############################################
    plt.xlabel('Number of traces')
    plt.ylabel('Guessing Entropy')
    plt.grid(True)
    axes = plt.gca()
    axes.set_ylim([0, 256])
    for i in range(len(rank_mean_y)):
        plt.plot(ranks_x[i][0], rank_mean_y[i], label="{} {}".format(name_models[i], n_settings[i]['title']),
                 marker=n_settings[i]['plot_marker'], color=plot_colors[i], markevery=0.1)
        plt.legend()

    i_counter = 0
    for model_name, model_settings in network_settings.items():
        for model_setting in model_settings:
            # Plot GE
            plt.figure()
            plt.xlabel('Number of traces')
            plt.ylabel('Guessing Entropy')
            plt.grid(True)
            axes = plt.gca()
            axes.set_ylim(y_limits[i_counter])
            axes.set_xlim(x_limits[i_counter])
            i_counter += 1

            plt.title("{} - {}".format(model_name, model_setting['title']))

            for i in range(len(model_setting['ge_x'])):
                plt.plot(model_setting['ge_x'][i], model_setting['ge_y'][i],
                         # label="{} - {}".format(model_name, model_setting['line_title'][i]),
                         label=f"Number layers {model_setting['num_layers'][i]}",
                         color=model_setting['plot_colors'][i])
            plt.legend()
            figure = plt.gcf()
            file_path = "/media/rico/Data/TU/thesis/report/img/cnn/rd/layers/vgg2/"
            file_name = f"{file_extension}_ge_VGGNumLayers2_k{model_setting['kernel_sizes'][0]}" \
                        f"_l2_{l2_penalty}{noise_string}.pdf"
            # figure.set_size_inches(16, 9)
            figure.savefig(f"{file_path}/{file_name}", dpi=100)

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":

    limits_x = [[0, 2000]] * 9
    limits_y = [[0, 140]] * 9
    plot_rd(1.0, limits_x, limits_y, False, file_extension="equal")

    limits_x = [[0, 450]] * 9
    limits_y = [[0, 130]] * 9
    plot_rd(0.75, limits_x, limits_y, False, file_extension="equal")

    limits_x = [[0, 90]] * 9
    limits_y = [[0, 90]] * 9
    plot_rd(0.5, limits_x, limits_y, False, file_extension="equal")

    limits_x = [[0, 20]] * 9
    limits_y = [[0, 55]] * 9
    plot_rd(0.25, limits_x, limits_y, False, file_extension="equal")

    limits_x = [[0, 10]] * 9
    limits_y = [[0, 50]] * 9
    plot_rd(0.0, limits_x, limits_y, False, file_extension="equal")



    # limits_x = [[-2, 25]] * 9
    # limits_y = [[-5, 70]] * 9
    # plot_rd(0.005, 0.5, limits_x, limits_y, False, file_extension="equal")


