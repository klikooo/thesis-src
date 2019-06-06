import util
import util_classes
import numpy as np

model_name = "VGGNumLayers"
kernel_sizes = [100, 50, 25, 20, 15,    # 1
                50, 25, 20, 15, 10,
                26, 20, 15, 10, 7,      # 3
                21, 15, 10, 7, 5,
                17, 15, 10, 5, 7, 3,    # 5
                17, 15, 10, 5, 7, 3,    # 6
                17, 15, 10, 5, 7, 3,    # 7
                17, 15, 10, 5, 7, 3,    # 8
                17, 15, 10, 5, 7, 3,    # 9
                100]
num_layers = [1] * 5
num_layers = num_layers + [2] * 5
num_layers = num_layers + [3] * 5
num_layers = num_layers + [4] * 5
num_layers = num_layers + [5] * 6
num_layers = num_layers + [6] * 6
num_layers = num_layers + [7] * 6
num_layers = num_layers + [8] * 6
num_layers = num_layers + [9] * 6
num_layers = num_layers + [7]
input_shape = 3500


load_function = util_classes.get_init_func(model_name)
params_list = np.zeros((len(kernel_sizes)))
names = []
ks = []
nl = []
for i in range(len(kernel_sizes)):
    args = {
        "input_shape": input_shape,
        "kernel_size": kernel_sizes[i],
        "channel_size": 32,
        "num_layers": num_layers[i],
        "n_classes": 256
    }
    model = load_function(args)
    num_params = util.count_parameters(model)
    print(f"{model_name}_k{kernel_sizes[i]:3}_l{num_layers[i]:2}={num_params}")

    params_list[i] = num_params
    names.append(f"{model_name}_k{kernel_sizes[i]}_c32_l{num_layers[i]}")
    ks.append(kernel_sizes[i])
    nl.append(num_layers[i])


threshold = 100000

kernel_map = {}
layer_map = {}
min_max = {}
for i in range(len(params_list)):
    x_min = params_list[i] - threshold
    x_max = params_list[i] + threshold
    k_sizes = []
    n_layers = []
    for j in range(len(params_list)):
        if x_min <= params_list[j] <= x_max:
            k_sizes.append(ks[j])
            n_layers.append(nl[j])
    kernel_map.update({names[i]: k_sizes})
    layer_map.update({names[i]: n_layers})
    min_max.update({names[i]: f"[{x_min}, {x_max}]"})

for k, v in kernel_map.items():
    print(k)
    print(f"kernels: {v} layers: {layer_map[k]} , min max: {min_max[k]}")
    print()

# print("std: {}".format(np.std(params_list)))
# print("mean {}".format(np.mean(params_list)))
