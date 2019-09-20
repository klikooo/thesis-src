from decimal import Decimal
import copy

import plots.spread.plot as plot
import matplotlib.pyplot as plt
import util

setting = {"experiment": '3',
           "data_set": util.DataSet.DPA_V4,
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


def plot_train_size(train_size, save_name, x_lim, y_lim, show=False, font_size=18):
    setting_spread = copy.deepcopy(setting)
    setting_spread.update({"network_name": "SpreadNet",
                           "line_title2": "$Spread_{PH}$",
                           "plot_colors": ["g"],
                           "train_size": train_size
                           })

    setting_dense = copy.deepcopy(setting)
    setting_dense.update({"network_name": "DenseNet",
                          "line_title2": "$MLP_{best}$",
                          "plot_colors": ["r"],
                          "spread_factor": 1,
                          "train_size": train_size
                          })

    setting_dense_spread = copy.deepcopy(setting)
    setting_dense_spread.update({"network_name": "DenseSpreadNet",
                                 "line_title2": "$MLP_{RT}$",
                                 "plot_colors": ["b"],
                                 "train_size": train_size})
    network_settings = {
        "SpreadNet": [setting_spread],
        "DenseNet": [setting_dense],
        "DenseSpreadNet": [setting_dense_spread]
    }
    plot.create_plot(network_settings, save_name, x_lim, y_lim, font_size=font_size)
    if show:
        plt.plot()


###############
# TEST FOR HW #
###############

# Set the global setting to HW
setting.update({"use_hw": True})

# Test for HW with different training sizes
path = "/media/rico/Data/TU/thesis/report/img/spread/DPAv4"
hw_save_name = f"{path}/hw_" + "{}.pdf"
plot_train_size(1000, hw_save_name.format(1000), [-1, 100], [0, 101], font_size=22)
plot_train_size(5000, hw_save_name.format(5000), [-1, 25], [0, 70])
plot_train_size(20000, hw_save_name.format(20000), [-1, 25], [0, 70])
plot_train_size(40000, hw_save_name.format(40000), [-1, 25], [0, 70])


###############
# TEST FOR ID #
###############

# Set the global setting to ID
setting.update({"use_hw": False})

# Test for ID with different training sizes
id_save_name = f"{path}/id_" + "{}.pdf"
plot_train_size(1000, id_save_name.format(1000), [-100, 3500], [0, 140])
plot_train_size(5000, id_save_name.format(5000), [-1, 25], [0, 70])
plot_train_size(20000, id_save_name.format(20000), [-1, 10], [0, 30])
plot_train_size(40000, id_save_name.format(40000), [-1, 10], [0, 20])

