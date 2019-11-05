from decimal import Decimal
import copy

import plots.spread.plot as plot
import matplotlib.pyplot as plt
import util

setting = {"experiment": '3',
           "data_set": util.DataSet.ASCAD,
           "subkey_index": 2,
           "unmask": True,
           "desync": 0,
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
           "plot_markers": ["*", "*", "+"]
           }


def plot_train_size(train_size, save_name, x_lim, y_lim, show=False):
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
    s_dense_spread = copy.deepcopy(setting)
    s_dense_spread.update({
        "plot_colors": ["black"],
        "experiment": '2',
        "line_title2": "$DenseBatch_{RT}$"
    })
    big_settings = {"plot_markers": ["H", "<", "*"],
                    "train_size": 40000}
    setting_dense_big = copy.deepcopy(setting_dense)
    setting_dense_big.update(big_settings)
    setting_dense_spread_big = copy.deepcopy(setting_dense_spread)
    setting_dense_spread_big.update(big_settings)
    s_dense_spread_big = copy.deepcopy(s_dense_spread)
    s_dense_spread_big.update(big_settings)

    network_settings = {
        "DenseNet": [setting_dense, setting_dense_big],
        "DenseSpreadNet": [setting_dense_spread, setting_dense_spread_big],
        "DenseBatch": [s_dense_spread, s_dense_spread_big]
    }
    # Update all training sizes
    for k in network_settings:
        for v in network_settings[k]:
            print(v)
            update_train_size(v)

    plot.create_plot(network_settings, save_name, x_lim, y_lim)
    if show:
        plt.plot()


def update_train_size(model_setting):
    new_title = model_setting['line_title2'] + " - {} traces".format(model_setting['train_size'])
    model_setting.update({"line_title2": new_title})


###############
# TEST FOR HW #
###############

# Set the global setting to HW
setting.update({"use_hw": True})

# Test for HW with different training sizes
path = "/media/rico/Data/TU/thesis/report/img/spread/compare"
hw_save_name = f"{path}/hw.pdf"
plot_train_size(1000, hw_save_name, [-1, 50], [0, 256])
# plot_train_size(5000, hw_save_name.format(5000), [-1, 5000], [0, 256])
# plot_train_size(20000, hw_save_name.format(20000), [-1, 5000], [0, 256])
# plot_train_size(40000, hw_save_name.format(40000), [-1, 50], [0, 256])


###############
# TEST FOR ID #
###############

# Set the global setting to ID
setting.update({"use_hw": False})

# Test for ID with different training sizes
id_save_name = f"{path}/id.pdf"
plot_train_size(1000, id_save_name, [-1, 50], [0, 256])
# plot_train_size(5000, id_save_name.format(5000), [-1, 5000], [0, 256])
# plot_train_size(20000, id_save_name.format(20000), [-1, 5000], [0, 256])
# plot_train_size(40000, id_save_name.format(40000), [-1, 50], [0, 256])

