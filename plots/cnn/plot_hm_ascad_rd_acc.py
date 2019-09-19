import plotly.graph_objects as go
import numpy as np
import json
import os

from plots.cnn.plot_heatmap import generate_annotations

hit_worst = False


def load_acc(l2_penal, unmasked, desync, hw, noise):
    noise_string = f"_noise{noise}" if noise > 0.0 else ''
    path = "/media/rico/Data/TU/thesis/runs3/" \
           "ASCAD_NORM/subkey_2/{}/{}/{}_SF1_E75_BZ100_LR1.00E-04{}_kaiming/train45000/".format(
            '' if unmasked else 'masked',
            f'desync{desync}' if desync > 0 else '',
            'HW' if hw else 'ID',
            '_L2_{}'.format(l2_penal) if l2_penal > 0 else '')
    print(path)
    model = "VGGNumLayers"

    acc_path = f"{path}/acc_{model}{noise_string}.json"
    with open(acc_path, "r") as f:
        return json.loads(f.read())


def write_to(ls, ks, l2_penal, noise_level, unmasked, hw, desync):
    noise_string = f"_noise{noise_level}" if noise_level > 0.0 else ''
    path = "/media/rico/Data/TU/thesis/runs3/" \
           "ASCAD_NORM/subkey_2/{}/{}/{}_SF1_E75_BZ100_LR1.00E-04{}_kaiming/train45000/".format(
            '' if unmasked else 'masked',
            f'desync{desync}' if desync > 0 else '',
            'HW' if hw else 'ID',
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


def do():
    todo = {
        50: {0.0, 0.1, 0.2, 0.3, 0.4, 0.5},
        100: {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
    }
    for desync, noise_levels in todo.items():
        for noise in noise_levels:
            layers = {1, 2, 3, 4, 5}
            kernels = {100, 50, 25, 20, 15, 10, 7, 5, 3}
            channels = 32
            l2_penal = 0.0
            unmask = True
            hw = True
            write_to(layers, kernels, l2_penal=l2_penal,
                     noise_level=noise, unmasked=unmask, hw=hw, desync=desync)
            data_acc = load_acc(l2_penal, unmask, desync, hw, noise)

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
            annotations = generate_annotations(sorted_data, x_labels, y_labels)

            color_worst = "#000000" if hit_worst else "#4CC01F"
            fig = go.Figure(data=go.Heatmap(
                z=sorted_data,
                x=x_labels,
                y=y_labels,
                colorbar={"title": "Accuracy (%)"}
            ))
            fig.update_layout(
                # title=f'Accuracy, l2 {l2_penal} noise {noise} desync {desync}',
                xaxis=go.layout.XAxis(
                    title=go.layout.xaxis.Title(text="Stacked layers per conv block"),
                    linecolor='black'
                ),
                yaxis=go.layout.YAxis(
                    title=go.layout.yaxis.Title(text="Kernel size"),
                    linecolor='black'
                ),
                annotations=annotations,
            )
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)
            fig.write_image(f"/media/rico/Data/TU/thesis/report/img/"
                            f"cnn/ascad_rd/hm/acc_desync{desync}_noise{noise}.pdf")


if __name__ == "__main__":
    do()
