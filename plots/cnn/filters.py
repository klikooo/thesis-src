from decimal import Decimal

import util
import util_classes
from models.load_model import load_model

import pdb

import matplotlib.pyplot as plt

settings = {"experiment": '3',
            "data_set": util.DataSet.RANDOM_DELAY_NORMALIZED,
            "subkey_index": 2,
            "unmask": True,
            "desync": 0,
            "use_hw": False,
            "spread_factor": 1,
            "epochs": 75,
            "batch_size": 100,
            "lr": '%.2E' % Decimal(0.0001),
            "l2_penalty": 0.005,
            "train_size": 40000,
            "kernel_size": 15,
            "num_layers": 1,
            "channel_size": 32,
            "network_name": "VGGNumLayers",
            "init_weights": "kaiming",
            "run": 0
            }
args = util.EmptySpace()
for key, value in settings.items():
    setattr(args, key, value)

folder = "/media/rico/Data/TU/thesis/runs{}/{}".format(args.experiment, util.generate_folder_name(args))
filename = folder + f"/model_r{args.run}_" + util_classes.get_save_name(args.network_name, settings) + ".pt"
model = load_model(args.network_name, filename)
print(model)

c = model.block3[0][0]
print(c)
# print(model.block1[0][0].weights)
w = c.weight
print(c.bias.size())
exit()
shape_weight = w.size()
print(f"Shape weight: {shape_weight}")

for channel_index in range(shape_weight[1]):
    plt.figure()
    for filter_index in range(shape_weight[0]):
        kernel = w[filter_index][channel_index].detach().cpu().numpy()

        plt.plot(abs(kernel))


    # while True:
    #     pdb.set_trace()
# for i in range(args.channel_size):

plt.show()
