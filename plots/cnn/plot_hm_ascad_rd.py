import plotly.graph_objects as go
import os
import util
import numpy as np

from plots.cnn.plot_heatmap import generate_annotations


def load_ge(kernel, l2_penal, desync, hw, unmask, noise, channels=32):
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
           "ASCAD_NORM/subkey_2/{}/{}/" \
           "{}_SF1_E75_BZ100_LR1.00E-04{}_kaiming/train45000/".format(
            '' if unmask else 'masked',
            f'desync{desync}' if desync > 0.0 else '',
            'HW' if hw else 'ID',
            '_L2_{}'.format(l2_penal) if l2_penal > 0 else '')
    print(path)
    model = "VGGNumLayers"

    all_ge = {}
    all_cge = {}
    for layers, kernel_sizes in combinations.items():
        kernel_size_dict = {}
        cge_dict = {}
        all_ge.update({layers: kernel_size_dict})
        all_cge.update({layers: cge_dict})
        for kernel_size in kernel_sizes:
            ge_runs = []
            noise_string = f"_noise{noise}" if noise > 0.0 else ''
            file = path + "/model_r{}_" + f"{model}_k{kernel_size}_c{channels}_l{layers}{noise_string}.exp"
            if not (os.path.exists(file.format(0)) or os.path.exists(file.format(0) + "__")):
                kernel_size_dict.update({kernel_size: float("nan")})
                cge_dict.update({kernel_size: float("nan")})
                continue
            sum_cge = 0.0
            num_runs = 5
            no_convergence = False
            for run in range(num_runs):
                filename = file.format(run)
                if os.path.exists(f"{filename}__"):
                    filename = f"{filename}__"
                ge_run = util.load_csv(filename, delimiter=' ', dtype=np.float)
                ge_runs.append(ge_run)
                run_cge = find_sequence(ge_run)
                if int(run_cge) == -100:
                    no_convergence = True
                sum_cge += run_cge

            mean_ge = np.mean(ge_runs, axis=0)
            cge = sum_cge/float(num_runs)
            if no_convergence:
                cge = -100
            kernel_size_dict.update({kernel_size: mean_ge})
            cge_dict.update({kernel_size: int(cge)})
    print(all_cge)
    return all_ge, all_cge


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
    todo = {
        50: {0.0, 0.1, 0.2, 0.3, 0.4, 0.5},
        100: {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
    }
    for desync, noise_levels in todo.items():
        for noise in noise_levels:

            kernels = {100, 50, 25, 20, 15, 10, 7, 5, 3}
            l2_penal = 0.0
            # desync = 50
            hw = True
            unmask = True
            # noise = 0.0
            data_ge, cge = load_ge(kernels, l2_penal, desync, hw, unmask, noise)
            minimal = get_first_min(data_ge)
            first = get_first(data_ge)
            end = get_end(data_ge)

            data_for_hm = minimal if unmask else end
            title = 'Convergence point' if unmask else 'Key rank after 10000 traces'

            x_labels = get_x_labels(minimal)
            y_labels = [f'K{i}' for i in sorted(list(kernels))]

            # z = np.transpose(get_sorted(data_for_hm))
            z = np.transpose(get_sorted(minimal))
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
                margin={
                    't': 5,
                    'b': 5
                }
            )
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)
            # fig.show()
            fig.write_image(f"/media/rico/Data/TU/thesis/report/img/"
                            f"cnn/ascad_rd/hm/ge_desync{desync}_noise{noise}.pdf")


if __name__ == "__main__":
    create()
