from decimal import Decimal
import copy

import plots.spread.plot as plot
import matplotlib.pyplot as plt
import util

setting = {"experiment": '',
           "data_set": util.DataSet.ASCAD,
           "subkey_index": 2,
           "unmask": True,
           "desync": 0,
           "use_hw": True,
           "spread_factor": 6,
           "epochs": 80,
           "batch_size": 100,
           "lr": '%.2E' % Decimal(0.0001),
           "l2_penalty": 0,
           "train_size": 1000,
           "kernel_sizes": [0],
           "num_layers": [0],
           "channel_sizes": [0],
           "network_name": "SpreadNet",
           "runs": range(5),
           "init_weights": "",
           "title": "",
           "plot_colors": ["acqua", "black", "brown", "darkblue", "darkgreen", "fuchsia",
                           "goldenrod", "green", "grey", "indigo", "lavender"],
           "ge_x": [],
           "ge_y": [],
           "ta": [],
           "va": [],
           "tl": [],
           "vl": [],
           "line_title": [],
           "line_title2": "$Spread_{PH}$",
           "plot_markers": [" ", "*", "+"]
           }


def plot_factors(spread_factors, save_name, x_lim, y_lim, show=False, train_size=1000, font_size=18,
                 change_run=''):
    setting_spread = copy.deepcopy(setting)
    setting_spread.update({"network_name": "SpreadNet",
                           "line_title2": "$Spread_{PH}$",
                           "plot_colors": ["r", "g", "b", "g"],
                           "plot_marker": [" "],
                           "train_size": train_size,
                           })

    setting_spread_norm = copy.deepcopy(setting)
    setting_spread_norm.update({"network_name": "DenseNorm",
                                "line_title2": "$Spread_{V2}$",
                                "plot_colors": ["r", "g", "b", "g"],
                                "plot_marker": [" "],
                                "train_size": train_size,
                                })

    setting_dense_batch = copy.deepcopy(setting)
    setting_dense_batch.update({"network_name": "DenseBatch",
                                "line_title2": "$DenseBatch_{RT}$",
                                "plot_colors": ["r", "g", "b", "g"],
                                "plot_marker": ["-"],
                                "train_size": train_size,
                                })

    setting_mlp_best = copy.deepcopy(setting)
    setting_mlp_best.update({"network_name": "DenseNet",
                             "line_title2": "$MLP_{BEST}$",
                             "plot_colors": ["r", "g", "b", "g"],
                             "plot_marker": ["<"],
                             "train_size": train_size,
                             })

    settings_spread_norm = []
    settings_spread = []
    settings_dense_batch = []
    colors = ["r", "g", "b", "y", "g", "b", "g", "r"]
    colors2 = ["y", "black", "pink"]
    i = 0
    for spread_factor, color in zip(spread_factors, colors):
        print(spread_factor)
        s_spread_norm = copy.deepcopy(setting_spread_norm)
        s_spread_norm.update({
            "spread_factor": spread_factor,
            "plot_colors": [color],
            "plot_markers": [" "],
            "experiment": change_run,
            "line_title2": s_spread_norm['line_title2'] + " sf " + str(spread_factor)
        })
        settings_spread_norm.append(s_spread_norm)

        s_spread = copy.deepcopy(setting_spread)
        s_spread.update({
            "experiment": '3',
            "spread_factor": spread_factor,
            "plot_colors": [color],
            "plot_markers": [">"],
            "line_title2": s_spread['line_title2'] + " sf " + str(spread_factor)
        })
        settings_spread.append(s_spread)

        s_dense_spread = copy.deepcopy(setting_dense_batch)
        s_dense_spread.update({
            "spread_factor": spread_factor,
            "plot_colors": [colors2[i]],
            "plot_markers": ["h"],
            "experiment": change_run,
            "line_title2": s_dense_spread['line_title2'] + " sf " + str(spread_factor)
        })
        settings_dense_batch.append(s_dense_spread)
        i += 1

    settings_mlp_best = []
    s_mlp_best = copy.deepcopy(setting_mlp_best)
    s_mlp_best.update({
        "experiment": '3',
        "spread_factor": 1,
        "plot_colors": ['silver'],
        "plot_markers": ["<"],
        "line_title2": s_mlp_best['line_title2']
    })
    settings_mlp_best.append(s_mlp_best)

    # TODO: Dense Batch should have spread factors right?
    network_settings = {
        "DenseNorm": settings_spread_norm, #Spread V2
        "DenseBatch": settings_dense_batch,
        # "SpreadNet": settings_spread, # Spread arch
        # "DenseNet": settings_mlp_best # MLP Best
    }
    plot.create_plot(network_settings, save_name, x_lim, y_lim, font_size=font_size, show_acc=False, show_loss=False)
    if show:
        plt.show()


#########
# ASCAD #
#########
data_set = util.DataSet.ASCAD
setting.update({"data_set": data_set})

###############
# TEST FOR HW #
###############

# Set the global setting to HW
setting.update({"use_hw": True})

# Test for HW with different training sizes
path = "/media/rico/Data/TU/thesis/report/img/spread/batch_norm"
hw_save_name = f"{path}/{data_set}_hw_" + "{}.pdf"
plot_factors([3, 6, 9], hw_save_name.format(1000), [0, 40], [0, 101], show=False, font_size=22)
plot_factors([3, 6, 9], hw_save_name.format(40000), [0, 40], [0, 101], show=False, font_size=22,
             train_size=40000, change_run='2')

# Set the global setting to ID
setting.update({"use_hw": False})

# Test for HW with different training sizes
path = "/media/rico/Data/TU/thesis/report/img/spread/batch_norm"
id_save_name = f"{path}/{data_set}_id_" + "{}.pdf"
plot_factors([3, 6, 9], id_save_name.format(1000), [0, 1000], [0, 140], show=False, change_run='2')

path = "/media/rico/Data/TU/thesis/report/img/spread/batch_norm"
id_save_name = f"{path}/{data_set}_id_" + "{}.pdf"
plot_factors([3, 6, 9], id_save_name.format(40000), [0, 10], [0, 20], show=False,
             train_size=40000, change_run='2')

