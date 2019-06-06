import numpy as np
from sklearn.preprocessing import StandardScaler

path = "/media/rico/Data/TU/thesis/data/Random_Delay/"
key_guesses_path = path + "Value/"
traces_path = path + "traces/"

traces_filename = traces_path + "traces_complete.csv.npy"


traces = np.load(traces_filename)

train_size = 40000
validation_size = 1000
test_size = 9000

x_train = traces[0:train_size]
x_validation = traces[train_size:train_size+validation_size]
x_test = traces[train_size+validation_size:train_size+validation_size+test_size]
print("loaded")

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_validation = scale.transform(x_validation)
x_test = scale.transform(x_test)

new_data = x_train
new_data = np.append(new_data, x_validation, axis=0)
new_data = np.append(new_data, x_test, axis=0)

normalized_traces_filename = traces_path + "traces_normalized_t{}_v{}.csv.npy".format(train_size, validation_size)
np.save(normalized_traces_filename, new_data)

# x = load_dataset_200(dataset, model)
#         length = len(x)
#         index_train_start = 0
#         index_validate_start = 50000
#         index_test_start = 55000
#         index_test_end = 60000
#
#         x_train = x[index_train_start:index_validate_start]     # train data
#         x_validate = x[index_validate_start:index_test_start]   # validation data
#         x_test = x[index_test_start:index_test_end]             # test data
#
#         scaler          = StandardScaler()
#         x_train         = scaler.fit_transform(x_train)
#         x_validate      = scaler.transform(x_validate)
#         x_test          = scaler.transform(x_test)
#
#         new_data = x_train
#         new_data = np.append(new_data, x_validate, axis=0)
#         new_data = np.append(new_data, x_test, axis=0)
#
#         np.save(datasets_dir + dataset + '_' + model + '_scaled.npy', new_data)