import numpy as np
import util

# Define paths and data set
path = "/media/rico/Data/TU/thesis/data/"
data_set = util.DataSet.RANDOM_DELAY_NORMALIZED

# Load
args = {
    "traces_path": path,
    "use_noise_data": False,
    "start": 0,
    "size": 50000,
    "data_set": util.DataSet.RANDOM_DELAY_NORMALIZED}
x_train, _, _ = util.load_data_generic(args)

# Create noise
sigma = 10
mu = 0
noise = np.random.normal(0, 1, 50000 * 3500) * 1.0
noise = noise.reshape(50000, 3500)
print(x_train.shape)
print(noise.shape)

# Add noise
noise_data = x_train + noise

# Simple check
print(f"x: {x_train[0][0]}\nn: {noise[0][0]}\nr: {noise_data[0][0]}")

# Save noise data
np.save(f"{path}/{str(data_set)}/traces/traces_noise", noise_data)

# Perform some tests if needed
# import pdb
# pdb.set_trace()





