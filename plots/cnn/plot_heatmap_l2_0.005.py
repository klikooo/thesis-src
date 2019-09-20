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
    l2_penal = 0.005

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
                filename = file.format(run)
                if os.path.exists(f"{filename}__"):
                    filename = f"{filename}__"
                ge_run = util.load_csv(filename, delimiter=' ', dtype=np.float)
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
        colorscale=[[0.0, "rgb(165,0,0)"],
                    [0.025, "rgb(185,10,10)"],
                    [0.05, "rgb(195,25,20)"],
                    [0.075, "rgb(205,40,30)"],
                    [0.1111111111111111, "rgb(215,48,39)"],
                    [0.2222222222222222, "rgb(244,109,67)"],
                    [0.3333333333333333, "rgb(253,174,97)"],
                    [0.4444444444444444, "rgb(254,224,144)"],
                    [0.5555555555555556, "rgb(224,243,248)"],
                    [0.6666666666666666, "rgb(171,217,233)"],
                    [0.7777777777777778, "rgb(116,173,209)"],
                    [0.8888888888888888, "rgb(69,117,180)"],
                    [1.0, "rgb(49,54,149)"]]
    ))
    fig.update_layout(
        title='Convergence point L2 0.005',
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(text="Stacked layers")
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text="Kernel size")
        ),
        margin={
            't': 5,
            'b': 5
        }
    )
    fig.show()
