import plotly.graph_objects as go
import os
import util
import numpy as np

from plots.cnn.plot_heatmap import generate_annotations

hit_worst = False


def load_ge(kernel, l2_penal, noise):
    combinations = {
        1: kernel,
        2: kernel,
        3: kernel,
        4: kernel,
        5: kernel,
        # 6: kernel,
        # 7: kernel,
        # 8: kernel,
        # 9: kernel,
    }

    path = "/media/rico/Data/TU/thesis/runs3/" \
           "Random_Delay_Normalized/subkey_2/ID_SF1_E75_BZ100_LR1.00E-04{}_kaiming/train40000/".format(
            '_L2_{}'.format(l2_penal) if l2_penal > 0 else '')
    print(path)
    model = "VGGNumLayers2"
    noise_string = f'_noise{noise}' if noise > 0 else ''

    all_ge = {}
    for layers, kernel_sizes in combinations.items():
        kernel_size_dict = {}
        all_ge.update({layers: kernel_size_dict})
        for kernel_size in kernel_sizes:
            ge_runs = []
            file = path + "/model_r{}_" + f"{model}_k{kernel_size}_c32_l{layers}_m4{noise_string}.exp"
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

            index = find_sequence(mean_ge)
            min_per_layer.update({kernel_size: index})
    return all_min


def find_sequence(data, epsilon=0.001, threshold=5, err=-100):
    global hit_worst
    joined = "".join(map(lambda x: '0' if x < epsilon else '1', data))
    index = joined.find("0" * threshold)
    if index == -1:
        hit_worst = True
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
    for noise in [0.0, 0.25, 0.5, 0.75, 1.0]:
        kernels = {100, 50, 25, 20, 15, 10, 7, 5, 3}
        l2_penal = 0.0
        data_ge = load_ge(kernels, l2_penal, noise)
        minimal = get_first_min(data_ge)
        first = get_first(data_ge)

        x_labels = get_x_labels(minimal)
        y_labels = [f'K{i}' for i in sorted(list(kernels))]

        print(get_sorted(first))

        color_worst = "#000000" if hit_worst else "#4CC01F"
        z = np.transpose(get_sorted(minimal))
        annotations = generate_annotations(z, x_labels, y_labels)

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorbar={"title": "CGE"},
            reversescale=True,
            colorscale='Viridis',
        ))
        fig.update_layout(
            # title=f'Convergence point, noise {noise}, L2 {l2_penal}',
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
                        f"cnn/rd/hm/vgg2/ge_noise{noise}.pdf")
