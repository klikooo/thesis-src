import json
import os
import util
import util_classes

data_set = util.DataSet.ASCAD_NORM


model = "VGGNumLayers"
noise = 0.0

kernels = [3, 20]
layers = [2]
channel_size = 32
max_pool = 5

desync = 100
masked_string = '/masked/'
hw_string = 'HW'
init_string = "_kaiming"
path = f"/media/rico/Data/TU/thesis/runs/{str(data_set)}/subkey_2/" \
       f"{masked_string}/desync{desync}/" \
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

        with open(f'{path}/{template_name}', 'r') as file:
            data = json.loads(file.read())
        acc_dict.update({full_model: data})
        print(f"{full_model}: {data}")

print(acc_dict)
acc_filename = f"{path}/acc_{model}.acc"
if os.path.exists(acc_filename):
    print(f"{util.BColors.FAIL}File already exists, skipping write to\n{acc_filename}{util.BColors.ENDC}")
    exit()
with open(acc_filename, "w") as file:
    file.write(json.dumps(acc_dict))
