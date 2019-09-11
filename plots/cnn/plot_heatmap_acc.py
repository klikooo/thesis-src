import plotly.graph_objects as go
import numpy as np
import json

from plots.cnn.plot_heatmap import generate_annotations

import os


def load_acc(l2_penal, noise_level):
    noise_string = f'_noise{noise_level}' if noise_level > 0.0 else ''
    path = "/media/rico/Data/TU/thesis/runs3/" \
           "Random_Delay_Normalized/subkey_2/ID_SF1_E75_BZ100_LR1.00E-04{}_kaiming/train40000/".format(
            '_L2_{}'.format(l2_penal) if l2_penal > 0 else '')
    print(path)
    model = "VGGNumLayers"

    acc_path = f"{path}/acc_{model}{noise_string}.json"
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


def write_to(ls, ks, l2_penal, noise_level):
    noise_string = f'_noise{noise_level}' if noise_level > 0.0 else ''
    path = "/media/rico/Data/TU/thesis/runs3/" \
           "Random_Delay_Normalized/subkey_2/ID_SF1_E75_BZ100_LR1.00E-04{}_kaiming/train40000/".format(
            '_L2_{}'.format(l2_penal) if l2_penal > 0 else '')

    filename = "acc_VGGNumLayers_k{}_c32_l{}" + noise_string + ".acc"
    data = {}
    for l in ls:
        for k in ks:
            file = path + filename.format(k, l)
            if os.path.exists(file):
                with open(file, "r") as f:
                    d = f.read()
                data.update({f'c_32_l{l}_k{k}': float(d)})
    new_filename = f"acc_VGGNumLayers{noise_string}.json"
    file = path + new_filename
    if os.path.exists(file):
        print("skipping creating acc")
        print(f"{file} exists, check it out. new content:")
        print(data)
        return
    with open(file, "w") as f:
        f.write(json.dumps(data))


if __name__ == "__main__":
    todo = {
        0.0: [0.0],
        0.05: [0.0, 0.25, 0.5, 0.75, 1.0],
        0.005: [0.0]
    }
    for l2_penal, noise_levels in todo.items():
        for noise_level in noise_levels:
            layers = {1, 2, 3, 4, 5, 6, 7, 8, 9}
            kernels = {100, 50, 25, 20, 15, 10, 7, 5, 3}
            channels = 32
            # l2_penal = 0.005
            # noise_level = 0.0
            write_to(layers, kernels, l2_penal, noise_level)
            data_acc = load_acc(l2_penal, noise_level)

            x_labels = [f'{i} ' for i in sorted(list(layers))]
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
            annotations = generate_annotations(sorted_data, x_labels, y_labels)
            fig = go.Figure(data=go.Heatmap(
                z=sorted_data,
                x=x_labels,
                y=y_labels,
            ))
            fig.update_layout(
                title=f'Accuracy, l2 {l2_penal}, noise {noise_level}',
                xaxis=go.layout.XAxis(
                    title=go.layout.xaxis.Title(text="Stacked layers per conv block"),
                    linecolor='black',
                    tickmode='linear',
                    # tickformat='%s'
                ),
                yaxis=go.layout.YAxis(
                    title=go.layout.yaxis.Title(text="Kernel size"),
                    linecolor='black',
                    # autorange=False
                    # dtick=1,
                    # tickformat='%s'
                ),
                annotations=annotations,
            )
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False, tickvals=y_labels)
            fig.write_image(f"/media/rico/Data/TU/thesis/report/img/"
                            f"cnn/rd/hm/acc_l2_{l2_penal}_noise{noise_level}.pdf")
            # fig.show()
            # exit()

