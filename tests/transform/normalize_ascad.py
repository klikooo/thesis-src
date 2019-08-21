import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py

import util

path = "/media/rico/Data/TU/thesis/data/"
path_ascad = path + "ASCAD_NORM/"

desync = 100
subkey = 2

traces_file = '{}/ASCAD{}.h5'.format(path_ascad,
                                     f'_desync{desync}' if desync > 0 else '')
traces = h5py.File(traces_file, 'r')
x_train_traces = traces['Profiling_traces/traces']
x_test_traces = traces['Attack_traces/traces']

train_size = 45000
validation_size = 1000
test_size = 10000

x_train = x_train_traces[0:train_size]
x_validation = x_train_traces[train_size:train_size + validation_size]
x_test = x_test_traces
print("loaded")

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_validation = scale.transform(x_validation)
x_test = scale.transform(x_test)

new_data = x_train
new_data = np.append(new_data, x_validation, axis=0)
new_data = np.append(new_data, x_test, axis=0)

# Save the traces
path_normalized = path + "ASCAD2/"
normalized_traces_filename = path_normalized + "traces/" + "traces_normalized_t{}_v{}_{}.csv.npy".format(
    train_size,
    validation_size,
    desync)
np.save(normalized_traces_filename, new_data)

# Only save the y values when we have desync 0
if desync == -1:
    # Save the masked y values
    y_train_masked = y_train_traces
    y_test_masked = y_test_traces
    y_filename_masked = path_normalized + "Value/model_masked_{}".format(subkey)

    y_masked = y_train_masked
    y_masked = np.append(y_masked, y_test_masked, axis=0)
    np.save(y_filename_masked, y_masked)

    y_train_unmasked = np.array([y_train_traces[i] ^ metadata_profiling[i]['masks'][subkey - 2] for i in range(50000)])
    y_test_unmasked = np.array([y_test_traces[i] ^ metadata_attack[i]['masks'][subkey - 2] for i in range(10000)])
    y_filename_unmasked = path_normalized + "Value/model_unmasked"

    y_unmasked = y_train_unmasked
    y_unmasked = np.append(y_unmasked, y_test_unmasked, axis=0)
    np.save(y_filename_unmasked, y_unmasked)
