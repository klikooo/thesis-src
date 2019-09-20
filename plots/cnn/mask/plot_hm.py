import plotly.graph_objects as go
import os
import util
import numpy as np

from plots.cnn.plot_heatmap import generate_annotations
from plots.cnn.plot_hm_ascad_rd import load_ge


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

            index = find_sequence(mean_ge)
            min_per_layer.update({kernel_size: index})
    return all_min


def find_sequence(data, epsilon=0.001, threshold=5, err=float("-100")):
    joined = "".join(map(lambda x: '0' if x < epsilon else '1', data))
    index = joined.find("0" * threshold)
    if index == -1:
        return err
    return index


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


def get_end(data):
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

            min_per_layer.update({kernel_size: mean_ge[9999]})
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


def create():
    kernels = {100, 50, 25, 20, 15, 10, 7, 5, 3}
    l2_penal = 0.0
    desync = 0
    hw = False
    unmask = False
    # noise = 0.0
    data_ge, cge = load_ge(kernels, l2_penal, desync, hw, unmask, 0)
    minimal = get_first_min(data_ge)
    first = get_first(data_ge)
    end = get_end(data_ge)

    title = 'Convergence point' if unmask else 'Key rank after 10000 traces'

    d = end

    x_labels = get_x_labels(d)
    y_labels = [f'K{i}' for i in sorted(list(kernels))]

    # z = np.transpose(get_sorted(data_for_hm))
    z = np.transpose(get_sorted(d))
    annotations = generate_annotations(z, x_labels, y_labels)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        reversescale=True,
        colorbar={"title": "CGE"}
    ))
    fig.update_layout(
        # title=f'{title}, unmask {unmask} hw {hw}, L2 {l2_penal}, desync {desync}, noise {noise}',
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(text="Stacked layers"),
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
    # fig.show()
    fig.write_image(f"/media/rico/Data/TU/thesis/report/img/"
                    f"cnn/ascad_masked/hm/ge.pdf")


if __name__ == "__main__":
    create()
