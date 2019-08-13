import plotly.graph_objects as go
import os
import util
import numpy as np


def load_ge():
    combinations = {
        1: {100, 50, 25, 20, 15, 10, 7, 5, 3},
        2: {100, 50, 25, 20, 15, 10, 7, 5, 3},
        3: {100, 50, 25, 20, 15, 10, 7, 5, 3},
        4: {100, 50, 25, 20, 15, 10, 7, 5, 3},
        5: {100, 50, 25, 20, 15, 10, 7, 5, 3},
        6: {100, 50, 25, 20, 15, 10, 7, 5, 3},
        7: {100, 50, 25, 20, 15, 10, 7, 5, 3},
        8: {100, 50, 25, 20, 15, 10, 7, 5, 3},
        9: {100, 50, 25, 20, 15, 10, 7, 5, 3}
    }
    l2_penal = 0.0

    path = "/media/rico/Data/TU/thesis/runs3/" \
           "Random_Delay_Normalized/subkey_2/ID_SF1_E75_BZ100_LR1.00E-04{}_kaiming/train40000/".format(
            '_L2_{}'.format(l2_penal) if l2_penal > 0 else '')
    print(path)
    model = "VGGNumLayers"

    all_ge = {}
    for layers, kernel_sizes in combinations.items():
        kernel_size_dict = {}
        all_ge.update({layers: kernel_size_dict})
        for kernel_size in kernel_sizes:
            ge_runs = []
            file = path + "/model_r{}_" + f"{model}_k{kernel_size}_c32_l{layers}.exp"
            if not (os.path.exists(file.format(0)) or os.path.exists(file.format(0) + "__")):
                kernel_size_dict.update({kernel_size: float("nan")})
                continue
            # if kernel_size == 3 and layers == 6:
            #     kernel_size_dict.update({kernel_size: float("nan")})
            #     continue
            for run in range(5):
                file = file.format(run)
                if os.path.exists(f"{file}__"):
                    file = f"{file}__"
                ge_run = util.load_csv(file, delimiter=' ', dtype=np.float)
                ge_runs.append(ge_run)
            mean_ge = np.mean(ge_runs, axis=0)
            kernel_size_dict.update({kernel_size: mean_ge})
    return all_ge


def get_first_min(data):
    all_min = {}
    for layers, kernel_sizes in data.items():
        min_per_layer = {}
        all_min.update({layers: min_per_layer})
        for kernel_size, mean_ge in kernel_sizes.items():
            # Check for NaN
            if not isinstance(mean_ge, np.ndarray) and np.isnan(mean_ge):
                min_per_layer.update({kernel_size: [mean_ge]})
                continue

            indices = np.where(mean_ge == 0.0)
            index = 3000
            print(np.shape(indices))
            if np.shape(indices) != (1, 0):
                index = indices[0][0]
            min_per_layer.update({kernel_size: index})
    return all_min


def get_first(data):
    all_min = {}
    for layers, kernel_sizes in data.items():
        min_per_layer = {}
        all_min.update({layers: min_per_layer})
        for kernel_size, mean_ge in kernel_sizes.items():
            print(mean_ge)
            # Check for NaN
            if not isinstance(mean_ge, np.ndarray) and np.isnan(mean_ge):
                min_per_layer.update({kernel_size: [mean_ge]})
                continue

            min_per_layer.update({kernel_size: mean_ge[0]})
    return all_min


def get_sorted(data):
    new_data = []
    for layer_key in sorted(data.keys()):
        li = []
        for kernel_key in sorted(data[layer_key].keys()):
            li.append(data[layer_key][kernel_key])
        new_data.append(li)
    return new_data


def get_x_labels(data):
    x_labels_ = []
    for layer_key in sorted(data.keys()):
        x_labels_.append(f"L{str(layer_key)}")
    return x_labels_


if __name__ == "__main__":
    data_ge = load_ge()
    minimal = get_first_min(data_ge)
    first = get_first(data_ge)

    x_labels = get_x_labels(minimal)
    y_labels = ['K3', 'K5', 'K7', 'K10', 'K15', 'K20', 'K25', 'K50', 'K100']

    print(get_sorted(first))
    fig = go.Figure(data=go.Heatmap(
        z=np.transpose(get_sorted(minimal)),
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0,  "#4CC01F"],
            [0.05, "#5EC321"],
            [0.1,  "#70C623"],
            [0.15, "#83C924"],
            [0.2,  "#96CD26"],
            [0.25, "#A9D028"],
            [0.3,  "#BCD32A"],
            [0.35, "#D0D52C"],
            [0.4,  "#D8CD2E"],
            [0.45, "#DBBF30"],
            [0.5,  "#DEB132"],
            [0.55, "#E19638"],
            [0.6,  "#E57C3D"],
            [0.65, "#E86343"],
            [0.7,  "#EB4C4A"],
            [0.75, "#ED506B"],
            [0.8,  "#F0568C"],
            [0.85, "#F25DAD"],
            [0.9,  "#F564CB"],
            [0.95, "#F76BE8"],
            [1,    "#EE72F8"]
        ],

    ))
    fig.update_layout(
        title='Convergence point',
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(text="Stacked layers")
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text="Kernel size")
        )
    )
    fig.show()
