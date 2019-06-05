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


def plot_factors(spread_factors, save_name, x_lim, y_lim, show=False, train_size=1000, font_size=18):
    setting_spread = copy.deepcopy(setting)
    setting_spread.update({"network_name": "DenseNorm",
                           "line_title2": "$SpreadV2_{PH}$",
                           "plot_colors": ["r", "g", "b"],
                           "plot_marker": [" "],
                           "train_size": train_size,
                           })

    setting_dense_spread = copy.deepcopy(setting)
    setting_dense_spread.update({"network_name": "DenseBatch",
                                 "line_title2": "$DenseBatch_{RT}$",
                                 "plot_colors": ["r", "g", "b"],
                                 "plot_marker": ["-"],
                                 "train_size": train_size,
                                 })
    settings_spread = []
    colors = ["r", "g", "b", "y", "g", "b"]
    for spread_factor, color in zip(spread_factors, colors):
        print(spread_factor)
        s_spread = copy.deepcopy(setting_spread)
        s_spread.update({
            "spread_factor": spread_factor,
            "plot_colors": [color],
            "plot_markers": [" "],
            "line_title2": s_spread['line_title2'] + " sf " + str(spread_factor)
        })
        settings_spread.append(s_spread)

    settings_dense_spread = []
    s_dense_spread = copy.deepcopy(setting_dense_spread)
    s_dense_spread.update({
        "spread_factor": 1,
        "plot_colors": ['black'],
        "plot_markers": ["h"],
        "line_title2": s_dense_spread['line_title2'] + " sf " + str(spread_factor)
    })
    settings_dense_spread.append(s_dense_spread)

    network_settings = {
        "DenseNorm": settings_spread,
        "DenseBatch": settings_dense_spread
    }
    plot.create_plot(network_settings, save_name, x_lim, y_lim, font_size=font_size)
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
hw_save_name = f"{path}/{data_set}_hw_" + "{}.png"
plot_factors([3, 6, 9], hw_save_name.format(1000), [-1, 40], [0, 101], show=True, font_size=22)


