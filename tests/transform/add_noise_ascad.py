import numpy as np
import util

# Define paths and data set
path = "/media/rico/Data/TU/thesis/data/"
data_set = util.DataSet.ASCAD_NORM

num_traces = 10000
num_features = 700
desync = 100
unmask = True

# Load
args = {
    "traces_path": path,
    "use_noise_data": False,
    "train_size": 45000,
    "validation_size": 1000,
    "start": 46000,
    "size": num_traces,
    "desync": desync,
    "unmask": unmask,
    "use_hw": False,
    "data_set": data_set}
x_train, _, _ = util.load_ascad_normalized(args)

# Create noise
noise_level = 0.4
sigma = 1
mu = 0
noise = np.random.normal(0, 1, num_traces * num_features) * noise_level
noise = noise.reshape(num_traces, num_features)
print(x_train.shape)
print(noise.shape)

# Add noise
noise_data = x_train + noise

# Simple check
print(f"x: {x_train[0][0]}\nn: {noise[0][0]}\nr: {noise_data[0][0]}")

# Save noise data
np.save(f"{path}/{str(data_set)}/traces/"
        f"traces_normalized_t45000_v1000_{desync}_noise{noise_level}.csv.npy", noise_data)

# Perform some tests if needed
# import pdb
# pdb.set_trace()





