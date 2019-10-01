import util
from plots.cnn.params.create_hm import create_hm
from util_classes import get_init_func
import numpy as np
import itertools


def count_params(model_name, data_set, scalar=1.0):
    channels = 32
    max_pool = 4
    all_kernels = [3, 5, 7, 10, 15, 20, 25, 50, 100]
    to_test = {
        1: [3, 5, 7, 10, 15, 20, 25, 50, 100],
        2: [3, 5, 7, 10, 15, 20, 25, 50, 100],
        3: [3, 5, 7, 10, 15, 20, 25, 50],
        4: [3, 5, 7, 10, 15, 20, 25],
        5: [3, 5, 7, 10, 15, 20],
        6: [3, 5, 7, 10, 15],
        7: [3, 5, 7, 10, 15],
        8: [3, 5, 7, 10, 15],
        9: [3, 5, 7, 10, 15],
    }

    hw = False if data_set == util.DataSet.RANDOM_DELAY_NORMALIZED else True
    input_shape = util.get_raw_feature_size(data_set)

    n_classes = 9 if hw else 256
    all_params = []
    for layer, kernels in to_test.items():
        layer_params = []
        for kernel in kernels:
            init_func = get_init_func(model_name)
            model = init_func({
                "kernel_size": kernel,
                "num_layers": layer,
                "channel_size": channels,
                "n_classes": n_classes,
                "input_shape": input_shape,
                "max_pool": max_pool,
            })
            num_params = util.count_parameters(model)
            layer_params.append(np.float64(num_params/scalar))

        for i in range(len(layer_params), len(all_kernels)):
            layer_params.append([float("nan")])
        all_params.append(layer_params)
        print(layer_params)

    y_labels = [f"K{size}" for size in all_kernels]
    x_labels = [f"L{size}" for size in to_test]
    return all_params, x_labels, y_labels


def do_all():
    sets = [util.DataSet.RANDOM_DELAY_NORMALIZED, util.DataSet.RANDOM_DELAY_NORMALIZED,
            util.DataSet.ASCAD_NORM]
    model_names = ["VGGNumLayers", "VGGNumLayers2", "VGGNumLayers"]
    paths = ["/media/rico/Data/TU/thesis/report/img/cnn/rd/hm/params_{}.pdf",
             "/media/rico/Data/TU/thesis/report/img/cnn/rd/hm/vgg2/params_{}.pdf",
             "/media/rico/Data/TU/thesis/report/img/cnn/ascad_rd/hm/params_{}.pdf"
             ]
    for model, data_set, path in zip(model_names, sets, paths):
        params, x_labels, y_labels = count_params(model_name=model,
                                                  data_set=data_set,
                                                  scalar=1000000.0)
        params = np.array(np.transpose(params))
        fig = create_hm(params, x_labels, y_labels,
                        color_bar_title="#Parameters x1M")

        fig.write_image(path.format(model))


if __name__ == "__main__":
    do_all()
