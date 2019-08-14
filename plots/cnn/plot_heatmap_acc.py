import plotly.graph_objects as go
import os
import util
import numpy as np
import json

hit_worst = False


def load_acc(l2_penal):

    path = "/media/rico/Data/TU/thesis/runs3/" \
           "Random_Delay_Normalized/subkey_2/ID_SF1_E75_BZ100_LR1.00E-04{}_kaiming/train40000/".format(
            '_L2_{}'.format(l2_penal) if l2_penal > 0 else '')
    print(path)
    model = "VGGNumLayers"

    acc_path = f"{path}/acc_{model}.json"
    with open(acc_path, "r") as f:
        return json.loads(f.read())


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

    # kernels = {i for i in range(5, 105, 5)}
    layers = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    kernels = {100, 50, 25, 20, 15, 10, 7, 5, 3}
    channels = 32
    l2_penal = 0.0
    data_acc = load_acc(l2_penal)

    x_labels = [f'L{i}' for i in sorted(list(layers))]
    y_labels = [f'K{i}' for i in sorted(list(kernels))]

    # Update the ones were we have no data of
    sorted_data = []
    for l in sorted(layers):
        k_list = []
        sorted_data.append(k_list)
        for k in sorted(kernels):
            key = f"c_{channels}_l{l}_k{k}"
            if key not in data_acc:
                data_acc.update({key: float("nan")})
            k_list.append(data_acc[key])
            print(f"l{l}k{k}: {data_acc[key]}")
    sorted_data = np.transpose(sorted_data)

    color_worst = "#000000" if hit_worst else "#4CC01F"
    fig = go.Figure(data=go.Heatmap(
        z=sorted_data,
        x=x_labels,
        y=y_labels,
        # colorscale=[
        #     [0.0,  color_worst],
        #     [0.05, "#5EC321"],
        #     [0.1,  "#70C623"],
        #     [0.15, "#83C924"],
        #     [0.2,  "#96CD26"],
        #     [0.25, "#A9D028"],
        #     [0.3,  "#BCD32A"],
        #     [0.35, "#D0D52C"],
        #     [0.4,  "#D8CD2E"],
        #     [0.45, "#DBBF30"],
        #     [0.5,  "#DEB132"],
        #     [0.55, "#E19638"],
        #     [0.6,  "#E57C3D"],
        #     [0.65, "#E86343"],
        #     [0.7,  "#EB4C4A"],
        #     [0.75, "#ED506B"],
        #     [0.8,  "#F0568C"],
        #     [0.85, "#F25DAD"],
        #     [0.9,  "#F564CB"],
        #     [0.95, "#F76BE8"],
        #     [1,    "#EE72F8"]
        # ],
    ))
    fig.update_layout(
        title='Accuracy',
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(text="Stacked layers"),
            linecolor='black'
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text="Kernel size"),
            linecolor='black'
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.show()
