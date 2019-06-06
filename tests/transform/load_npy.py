import numpy as np
import util

path = "/media/rico/Data/TU/thesis/data/Random_Delay/"
traces_path = path + "traces/"
traces_filename = traces_path + "traces_complete.csv.npy"

model_path = path + "Value/"
model_filename = model_path + "model.csv.npy"
model_csv_filename = model_path + "model.csv"

x = np.load(model_filename)
x = x.reshape((50000,))
print(x.shape)
print(x)
print("mem usage {}".format(util.format_bytes(util.get_memory())))

y_train = util.load_csv(model_csv_filename,
                        delimiter=' ',
                        dtype=np.long,
                        start=0,
                        size=50000)

print(y_train.shape)
print(y_train)
for i in range(len(y_train)):
    if x[i] != y_train[i]:
        print("Error at {}".format(i))

