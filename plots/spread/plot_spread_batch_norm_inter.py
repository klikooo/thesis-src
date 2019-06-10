from decimal import Decimal

import util
import util_classes
from models.load_model import load_model

import numpy as np
import pdb
import torch

settings = {"experiment": '',
            "data_set": util.DataSet.ASCAD,
            "subkey_index": 2,
            "unmask": True,
            "desync": 0,
            "use_hw": True,
            "spread_factor": 6,
            "epochs": 80,
            "batch_size": 100,
            "lr": '%.2E' % Decimal(0.0001),
            "l2_penalty": 0,
            "train_size": 1000,
            "kernel_size": 20,
            "num_layers": 2,
            "channel_size": 16,
            "network_name": "SpreadV3", #""DenseNorm",
            "init_weights": "",
            "run": 0
}

args = util.EmptySpace()
for key, value in settings.items():
    setattr(args, key, value)

folder = "/media/rico/Data/TU/thesis/runs{}/{}".format(args.experiment, util.generate_folder_name(args))
filename = folder + f"/model_r{args.run}_" + util_classes.get_save_name(args.network_name, settings) + ".pt"
model = load_model(args.network_name, filename)

print(model)

x_test, _, _, _, _ = util.load_ascad_test_traces({
    "sub_key_index": 2,
    "desync": 0,
    "traces_path": "/media/rico/Data/TU/thesis/data",
    "unmask": args.unmask,
    "use_hw": args.use_hw
})
x_test = x_test
print(f"Shape x_test {np.shape(x_test)}")
x_test = torch.from_numpy(x_test.astype(np.float32)).to(util.device)


start = 0
for i in range(int(10000/100)):
    model(x_test[start:start+100])
    start += 100


print(f"Shape intermediate {np.shape(model.intermediate_values)}")

z = model.intermediate_values
res = np.zeros(600)
for batch_index in range(np.shape(z)[0]):
    batch = z[batch_index]
    s = np.sum(batch, axis=0)
    print(f"Shape s: {np.shape(s)}")
    res += s

print(res)
print(f"Number non zero: {np.count_nonzero(res)}")
# pdb.set_trace()
