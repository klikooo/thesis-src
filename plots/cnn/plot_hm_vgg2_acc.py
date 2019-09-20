import plotly.graph_objects as go
import numpy as np
import json

from plots.cnn.plot_heatmap import generate_annotations

hit_worst = False


def load_acc(l2_penal, noise_level):
    path = "/media/rico/Data/TU/thesis/runs3/" \
           "Random_Delay_Normalized/subkey_2/ID_SF1_E75_BZ100_LR1.00E-04{}_kaiming/train40000/".format(
            '_L2_{}'.format(l2_penal) if l2_penal > 0 else '')
    print(path)
    model = "VGGNumLayers2"

    acc_path = f"{path}/acc_{model}_noise{float(noise_level)}.json"
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

    for noise in [0.0, 0.25, 0.5, 0.75, 1.0]:

        layers = {1, 2, 3, 4, 5}
        kernels = {100, 50, 25, 20, 15, 10, 7, 5, 3}
        channels = 32
        l2_penal = 0.0
        data_acc = load_acc(l2_penal, noise)

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

        fig = go.Figure(data=go.Heatmap(
            z=sorted_data,
            x=x_labels,
            y=y_labels,
            colorbar={"title": "Accuracy (%)"},
            reversescale=False,
            colorscale='Viridis',
        ))
        fig.update_layout(
            # title=f'Accuracy, l2 {l2_penal}, noise {float(noise)}',
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(text="Stacked layers"),
                linecolor='black'
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(text="Kernel size"),
                linecolor='black'
            ),
            annotations=annotations,
            margin={
                't': 5,
                'b': 5
            }
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        fig.write_image(f"/media/rico/Data/TU/thesis/report/img/"
                        f"cnn/rd/hm/vgg2/acc_noise{noise}.pdf")
