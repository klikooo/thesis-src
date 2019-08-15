import plotly.graph_objects as go
import numpy as np
import util
import util_classes


def load_model(model_name, c, k, l):
    args = {
        "kernel_size": k,
        "channel_size": c,
        "num_layers": l,
        "n_classes": 256,
        "input_shape": 3500
    }
    init_func = util_classes.get_init_func(model_name)
    return init_func(args)


def all_count_params(model_name, num_layers, kernel_sizes, channel_size):
    d = []
    for l in sorted(num_layers):
        k_list = []
        d.append(k_list)
        for k in sorted(kernel_sizes):
            if l > 5 and k > 15:
                k_list.append(float('nan'))
                continue
            elif l == 5 and k > 20:
                k_list.append(float('nan'))
                continue
            elif l == 4 and k > 25:
                k_list.append(float('nan'))
                continue
            elif l == 3 and k > 50:
                k_list.append(float('nan'))
                continue

            model = load_model(model_name, channel_size, k, l)
            params = util.count_parameters(model)
            k_list.append(params)
    return d


if __name__ == "__main__":
    # kernels = {i for i in range(5, 105, 5)}
    layers = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    kernels = {100, 50, 25, 20, 15, 10, 7, 5, 3}
    channels = 32

    x_labels = [f'L{i}' for i in sorted(list(layers))]
    y_labels = [f'K{i}' for i in sorted(list(kernels))]
    sorted_data = np.transpose(all_count_params("VGGNumLayers", layers, kernels, channels))
    print(sorted_data)

    fig = go.Figure(data=go.Heatmap(
        z=sorted_data,
        x=x_labels,
        y=y_labels,
    ))
    fig.update_layout(
        title='Number of parameters',
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
