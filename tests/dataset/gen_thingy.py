import util
import numpy as np

traces_path = "/media/rico/Data/TU/thesis/data"
leakage_model = "HW"
path = '{}/DPAv4/traces/traces_50_{}.csv'.format(traces_path, leakage_model)
start = 0
size = 100000
save_to = f"{traces_path}/DPAv4/traces/traces_50_{leakage_model}.npy"

x_train = util.load_csv(path,
                        delimiter=' ',
                        start=start,
                        size=size)

print(np.shape(x_train))
np.save(save_to, np.array(x_train))


