import numpy as np
import util

path = "/media/rico/Data/TU/thesis/data/Random_Delay/"
traces_path = path + "traces/"
traces_filename = traces_path + "traces_complete.csv"
new_traces_filename = traces_path + "traces_complete1.npy"

x = np.load(new_traces_filename)
print(x.shape)
print(x)
print("mem usage {}".format(util.format_bytes(util.get_memory())))