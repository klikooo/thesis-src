import json
import os
import util
import util_classes

data_set = util.DataSet.ASCAD_NORM


model = "VGGNumLayers"
noise = 0.0

kernels = [100, 50, 25, 20, 15, 10, 7, 5, 3]
layers = [1, 2, 3, 4, 5]
channel_size = 32
max_pool = 5

desync = 100
masked_string = '/masked/' if True else '/'
hw_string = 'HW' if False else 'ID'
init_string = "_kaiming"
desync_string = f"desync{desync}" if desync > 0 else ''
path = f"/media/rico/Data/TU/thesis/runs3/{str(data_set)}/subkey_2/" \
       f"{masked_string}/{desync_string}/" \
       f"{hw_string}_SF1_E75_BZ100_LR1.00E-04{init_string}/train45000/"


acc_dict = {}
for kernel in kernels:
    for layer in layers:

        full_model = util_classes.get_save_name(model, {
            "kernel_size": kernel,
            "num_layers": layer,
            "channel_size": channel_size,
            "max_pool": max_pool
        })

        template_name = f"acc_{full_model}{f'_noise{noise}' if noise > 0.0 else ''}.acc"
        f = f'{path}/{template_name}'
        if not os.path.exists(f):
            print(f"{util.BColors.WARNING} {template_name} does not exists{util.BColors.ENDC}")
            continue

        with open(f, 'r') as file:
            data = json.loads(file.read())
        params = f"c_{channel_size}_l{layer}_k{kernel}"
        acc_dict.update({params: data})
        print(f"{full_model}: {data}")

print(acc_dict)
acc_filename = f"{path}/acc_{model}.acc"
if os.path.exists(acc_filename):
    print(f"{util.BColors.FAIL}File already exists, skipping write to\n{acc_filename}{util.BColors.ENDC}")
    exit()
with open(acc_filename, "w") as file:
    file.write(json.dumps(acc_dict))
