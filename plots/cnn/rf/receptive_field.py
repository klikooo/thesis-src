import math
import numpy  as np

from plots.cnn.params.create_hm import create_hm


def get_field_size(kernel_size, num_layers, max_pool=2, input_shape=3500):
    padding = int(kernel_size/2)
    stride = 1
    pool = [max_pool, max_pool, 0]
    z = [[kernel_size, stride, padding]] * num_layers
    z_labels = ['conv'] * num_layers
    pool_label = 'pool'
    convnet = [*z, pool, *z, pool, *z, pool]
    layer_names = [*z_labels, pool_label, *z_labels, pool_label, *z_labels, pool_label]
    imsize = input_shape


    def outFromIn(conv, layerIn):
        n_in = layerIn[0]
        j_in = layerIn[1]
        r_in = layerIn[2]
        start_in = layerIn[3]
        k = conv[0]
        s = conv[1]
        p = conv[2]

        n_out = math.floor((n_in - k + 2 * p) / s) + 1
        actualP = (n_out - 1) * s - n_in + k
        pR = math.ceil(actualP / 2)
        pL = math.floor(actualP / 2)

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - pL) * j_in
        return n_out, j_out, r_out, start_out


    def printLayer(layer, layer_name):
        print(layer_name + ":")
        print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (
        layer[0], layer[1], layer[2], layer[3]))


    # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    print("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, "input image")
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        # layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])
    print("------------------------")
    return currentLayer[2]


def get_all_fields():
    all_kernels = [3, 5, 7, 10, 15, 20, 25, 50, 100]
    settings = {
        1: [3, 5, 7, 10, 15, 20, 25, 50, 100],
        2: [3, 5, 7, 10, 15, 20, 25, 50, 100],
        3: [3, 5, 7, 10, 15, 20, 25, 50],
        4: [3, 5, 7, 10, 15, 20, 25],
        5: [3, 5, 7, 10, 15, 20],
    }
    x_labels = [f'L{l}' for l in settings]
    y_labels = [f'K{l}' for l in all_kernels]

    extra_settings = [
        (2, 3500, "/media/rico/Data/TU/thesis/report/img/cnn/rd/hm/rf_VGGNumLayers.pdf"),
        (4, 3500, "/media/rico/Data/TU/thesis/report/img/cnn/rd/hm/vgg2/rf_VGGNumLayers2.pdf"),
        (2, 700, "/media/rico/Data/TU/thesis/report/img/cnn/ascad_rd/hm/rf_VGGNumLayers.pdf")
    ]
    for (max_pool, input_shape, path) in extra_settings:
        data = []
        for layers, kernels in settings.items():
            rfs = []
            for kernel in kernels:
                rf = get_field_size(kernel, layers, max_pool=max_pool, input_shape=input_shape)
                rfs.append(int(rf))
            for i in range(len(rfs), len(all_kernels)):
                rfs.append([float("nan")])
            data.append(rfs)

        data = np.transpose(data).tolist()
        fig = create_hm(data, x_labels, y_labels, color_bar_title="Receptive Field")
        fig.write_image(path)
        # fig.show()


if __name__ == "__main__":
    get_all_fields()
