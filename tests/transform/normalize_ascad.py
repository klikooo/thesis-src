import numpy as np
from sklearn.preprocessing import StandardScaler

import util

path = "/media/rico/Data/TU/thesis/data/ASCAD/"
key_guesses_path = path + "Value/"
traces_path = path + "traces/"


traces_file = '{}/ASCAD_{}_desync{}.h5'.format(path, 2, 0)
print('Loading {}'.format(traces_file))
(x_train_traces, y_train_traces), (x_test_traces, y_test_traces), (metadata_profiling, metadata_attack) = \
    util.load_ascad(traces_file, load_metadata=True)


train_size = 45000
validation_size = 1000
test_size = 10000

x_train = x_train_traces[0:train_size]
x_validation = x_train_traces[train_size:train_size+validation_size]
x_test = x_test_traces
print("loaded")

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_validation = scale.transform(x_validation)
x_test = scale.transform(x_test)

new_data = x_train
new_data = np.append(new_data, x_validation, axis=0)
new_data = np.append(new_data, x_test, axis=0)

normalized_traces_filename = path + "traces_normalized_t{}_v{}.csv.npy".format(train_size, validation_size)
np.save(normalized_traces_filename, new_data)


