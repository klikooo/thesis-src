import itertools
from decimal import Decimal

import util
import numpy as np

import matplotlib.pyplot as plt

from util_classes import get_save_name
import os

import matplotlib
from matplotlib.lines import Line2D

matplotlib.rcParams.update({'font.size': 18})


def plot_ascad(hw, desync, noise_level, x_limits, y_limits, show=True, file_extension=""):
    #####################################################################################
    # Parameters
    use_hw = hw
    spread_factor = 1
    runs = [x for x in range(5)]
    train_size = 45000
    epochs = 75
    batch_size = 100
    lr = 0.0001
    sub_key_index = 2

    unmask = True  # False if sub_kezy_index < 2 else True
    kernel_sizes = []
    num_layers = []
    channel_sizes = [32]
    init_weights = "kaiming"
    l2_penalty = 0.0

    network_1 = "VGGNumLayers"
    network_settings = {
        network_1: 5,
    }
    data_set = util.DataSet.ASCAD_NORM
    load_loss_acc = True
    show_acc = False
    experiment = False
    show_loss = False
    show_per_layer = True
    colors = ["aqua", "black", "brown", "darkblue", "darkgreen",
              "fuchsia", "goldenrod", "grey", "indigo", "lavender"]
    plot_markers = [" ", "*", ".", "o", "+", "8", "s", "p", "P", "h", "H"]
    hw_string = 'HW' if use_hw else 'ID'
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
    network_settings[network_1][0].update({
        "kernel_sizes": [100, 50, 25, 20, 15, 10, 7, 5, 3],
        "num_layers": [1] * 9,
        "title": " 1 layers desync {}, {}".format(desync, hw_string),
        "plot_marker": " ",
    })
    network_settings[network_1][1].update({
        "kernel_sizes": [100, 50, 25, 20, 15, 10, 7, 5, 3],
        "num_layers": [2] * 9,
        "title": " 2 layers desync {}, {}".format(desync, hw_string),
        "plot_marker": "*",

    })
    network_settings[network_1][2].update({
        "kernel_sizes": [50, 25, 20, 15, 10, 7, 5, 3],
        "num_layers": [3] * 8,
        "title": " 3 layers desync {}, {}".format(desync, hw_string),
        "plot_marker": ".",

    })
    network_settings[network_1][3].update({
        "kernel_sizes": [25, 20, 15, 10, 7, 5, 3],
        "num_layers": [4] * 7,
        "title": " 4 layers desync {}, {}".format(desync, hw_string),
        "plot_marker": "o",

    })
    network_settings[network_1][4].update({
        "kernel_sizes": [20, 15, 10, 7, 5, 3],
        "num_layers": [5] * 6,
        "title": " 5 layers desync {}, {}".format(desync, hw_string),
        "plot_marker": "+",

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
            ge_path = '{}{}.exp__'.format(filename, f'_noise{noise_level}' if noise_level > 0.0 else '')
            if not os.path.exists(ge_path):
                ge_path = f"{filename}{f'_noise{noise_level}' if noise_level > 0.0 else ''}.exp"

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
    plt.xlabel('Number of traces', fontsize=16)
    plt.ylabel('Guessing Entropy', fontsize=16)
    plt.grid(True)
    axes = plt.gca()
    axes.set_ylim([0, 256])
    for i in range(len(rank_mean_y)):
        plt.plot(ranks_x[i][0], rank_mean_y[i], label="{} {}".format(name_models[i], n_settings[i]['title']),
                 marker=n_settings[i]['plot_marker'], color=plot_colors[i], markevery=0.1)
        plt.legend()

    noise_string = f'_noise{noise_level}' if noise_level > 0.0 else ''
    if show_per_layer:
        validation_marker = "H"
        training_marker = " "
        for model_name, model_settings in network_settings.items():
            for model_setting in model_settings:
                line_marker = itertools.cycle((' ', '+', '<', 'o', "D", "H", "*", "."))
                ks_training_loss = model_setting['ta']
                ks_training_acc = model_setting['tl']
                ks_validation_acc = model_setting['vl']
                ks_validation_loss = model_setting['va']

                # Show loss
                ks = model_setting['kernel_sizes']
                labels = [f"Kernel size {k}" for k in ks]

                iter_colors = itertools.cycle(colors)
                line_labels = [Line2D([0], [0], color=next(iter_colors), lw=2) for _ in ks]
                # line_marker = itertools.cycle((' ', '+', '<', 'o', "D", "H", "*", "."))

                # SHOW LOSS
                fig, ax = plt.subplots()
                ax.legend([Line2D([0], [0], color='black', lw=2, marker=training_marker),
                           Line2D([0], [0], color='black', lw=2, marker=validation_marker),
                           *line_labels],
                          ['Training', 'Validation', *labels])
                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title("Loss {} - {}".format(model_name, model_setting['title']))
                iter_colors = itertools.cycle(colors)
                for i in range(len(ks_validation_loss)):
                    color = next(iter_colors)
                    plt.plot(ks_validation_loss[i],
                             marker=validation_marker,
                             color=color, markevery=0.1)
                    plt.plot(ks_training_loss[i],
                             marker=training_marker,
                             color=color, markevery=0.1)

                file_path = "/media/rico/Data/TU/thesis/report/img/cnn/ascad_rd/loss"
                file_name = f"loss_{hw_string}_VGGNumLayers_layers_{model_setting['num_layers'][0]}" \
                            f"_desync{desync}{noise_string}.png"
                figure = plt.gcf()
                figure.set_size_inches(16, 9)
                figure.savefig(f"{file_path}/{file_name}", dpi=100)

                # SHOW ACCURACY
                fig, ax = plt.subplots()
                ax.legend([Line2D([0], [0], color='black', lw=2, marker=training_marker),
                           Line2D([0], [0], color='black', lw=2, marker=validation_marker),
                           *line_labels],
                          ['Training', 'Validation', *labels])
                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title("Accuracy {} - {}".format(model_name, model_setting['title']))
                iter_colors = itertools.cycle(colors)
                for i in range(len(ks_validation_acc)):
                    color = next(iter_colors)
                    plt.plot(ks_validation_acc[i],
                             marker=validation_marker,
                             color=color, markevery=0.1)
                    plt.plot(ks_training_acc[i],
                             marker=training_marker,
                             color=color, markevery=0.1)

                file_path = "/media/rico/Data/TU/thesis/report/img/cnn/ascad_rd/acc"
                file_name = f"acc_{hw_string}_VGGNumLayers_layers_{model_setting['num_layers'][0]}" \
                            f"_desync{desync}{noise_string}.png"
                figure = plt.gcf()
                figure.set_size_inches(16, 9)
                figure.savefig(f"{file_path}/{file_name}", dpi=100)

    i_counter = 0
    for model_name, model_settings in network_settings.items():
        for model_setting in model_settings:
            # Plot GE
            plt.figure()
            plt.xlabel('Number of traces', fontsize=16)
            plt.ylabel('Guessing Entropy', fontsize=16)
            plt.grid(True)
            axes = plt.gca()
            axes.set_ylim(y_limits[i_counter])
            axes.set_xlim(x_limits[i_counter])
            i_counter += 1

            plt.title("{} - {}".format(model_name, model_setting['title']))

            for i in range(len(model_setting['ge_x'])):
                plt.plot(model_setting['ge_x'][i], model_setting['ge_y'][i],
                         # label="{} - {}".format(model_name, model_setting['line_title'][i]),
                         label=f"Kernel size {model_setting['kernel_sizes'][i]}",
                         color=model_setting['plot_colors'][i])
            plt.legend()
            figure = plt.gcf()
            file_path = "/media/rico/Data/TU/thesis/report/img/cnn/ascad_rd"
            file_name = f"{file_extension}_ge_{hw_string}_VGGNumLayers_layers_" \
                        f"{model_setting['num_layers'][0]}_desync{desync}{noise_string}.png"
            figure.set_size_inches(16, 9)
            figure.savefig(f"{file_path}/{file_name}", dpi=100)

            # Plot accuracy if asked for
            if show_acc:
                plt.figure()
                plt.title("Accuracy during training {} - {}".format(model_name, model_setting['title']))
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.grid(True)
                for i in range(len(model_setting['ge_x'])):
                    plt.plot(model_setting['ta'][i] * 100, label="Train {}".format(model_setting['line_title'][i]),
                             color='orange', marker=plot_markers[i], markevery=0.1)
                    plt.plot(model_setting['va'][i] * 100, label="Train {}".format(model_setting['line_title'][i]),
                             color='green', marker=plot_markers[i], markevery=0.1)
                plt.legend()

            if show_loss:
                plt.figure()
                plt.title("Loss during training {} - {}".format(model_name, model_setting['title']))
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                for i in range(len(model_setting['ge_x'])):
                    plt.plot(model_setting['tl'][i] * 100, label="Train {}".format(model_setting['line_title'][i]),
                             color='orange', marker=plot_markers[i], markevery=0.1)
                    plt.plot(model_setting['vl'][i] * 100, label="Train {}".format(model_setting['line_title'][i]),
                             color='green', marker=plot_markers[i], markevery=0.1)
                plt.legend()
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    ########################
    # PLOT WITH EQUAL AXES #
    ########################
    # limits_x = [[-2, 1000]] * 5
    # limits_y = [[-5, 128]] * 5
    # plot_ascad(hw=False, desync=50, noise_level=0.0, x_limits=limits_x, y_limits=limits_y,
    #            show=False, file_extension="equal")
    #
    # limits_x = [[-2, 20]] * 5
    # limits_y = [[-5, 70]] * 5
    # plot_ascad(hw=True, desync=50, noise_level=0.0, x_limits=limits_x, y_limits=limits_y,
    #            show=False, file_extension="fitting")

    limits_x = [[-2, 1000]] * 5
    limits_y = [[-5, 250]] * 5
    plot_ascad(hw=True, desync=50, noise_level=0.25, x_limits=limits_x, y_limits=limits_y,
               show=False, file_extension="fitting")

    limits_x = [[-2, 10000]] * 5
    limits_y = [[-5, 160]] * 5
    plot_ascad(hw=True, desync=50, noise_level=0.5, x_limits=limits_x, y_limits=limits_y,
               show=False, file_extension="fitting")

    # limits_x = [[-2, 1000]] * 5
    # limits_y = [[-5, 250]] * 5
    # plot_ascad(hw=True, desync=50, noise_level=0.75, x_limits=limits_x, y_limits=limits_y,
    #            show=False, file_extension="fitting")

    # limits_x = [[-2, 1000]] * 5
    # limits_y = [[-5, 250]] * 5
    # plot_ascad(hw=True, desync=50, noise_level=1.0, x_limits=limits_x, y_limits=limits_y,
    #            show=False, file_extension="fitting")

    ###############################
    # PLOT WITH GOOD FITTING AXES #
    # ###############################
    # limits_x = [[-2, 400], [-2, 1500], [-2, 400], [-2, 400], [-2, 400], [-2, 400], [-2, 400], [-2, 400], [-2, 400]]
    # limits_y = [[-5, 128], [-5, 128], [-5, 128], [-5, 128], [-5, 128], [-5, 128], [-5, 128], [-5, 128], [-5, 128]]
    # plot_ascad(0, limits_x, limits_y, show=False, file_extension="fitting")
