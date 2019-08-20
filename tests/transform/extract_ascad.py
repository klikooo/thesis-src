import h5py
import numpy as np
import util
import pdb
from sklearn.preprocessing import StandardScaler
import csv


path = "/media/rico/Data/TU/thesis/data/"
path_ascad = path + "ASCAD_Keys/"

desync = 0
subkey = 2

ascad_db = '{}/ascad.h5'.format(path_ascad)
in_file = h5py.File(ascad_db, "r")


# Training data
train_size = 200000
train_metadata = in_file['Profiling_traces/metadata']
x_train_traces = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
y_train_unmasked_all = np.array(in_file['Profiling_traces/labels'])

# Unmasked
y_train_unmasked = np.array([y_train_unmasked_all[i] for i in range(train_size)])
y_train_hw_unmasked = np.array([util.HW[y_train_unmasked[i]] for i in range(train_size)])

# Masked
y_train_masked = np.array([y_train_unmasked[i] ^ train_metadata[i]['masks'][subkey - 2] for i in range(train_size)])
y_train_hw_masked = np.array([util.HW[y_train_masked[i]] for i in range(train_size)])

# Keys + plaintext
train_keys = np.array([train_metadata[i]['key'][subkey] for i in range(train_size)])
train_plaintexts = np.array([train_metadata[i]['plaintext'][subkey] for i in range(train_size)])

# Test data
test_size = 1000
test_metadata = in_file['Attack_traces/metadata']
x_test_traces = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
y_test_unmasked_all = np.array(in_file['Attack_traces/labels'])

# Unmasked
y_test_unmasked = np.array([y_test_unmasked_all[i] for i in range(test_size)])
y_test_hw_unmasked = np.array([util.HW[y_test_unmasked[i]] for i in range(test_size)])

# Masked
y_test_masked = np.array([y_test_unmasked[i] ^ test_metadata[i]['masks'][subkey - 2] for i in range(test_size)])
y_test_hw_masked = np.array([util.HW[y_test_masked[i]] for i in range(test_size)])

# Keys + plaintext
test_keys = np.array([test_metadata[i]['key'][subkey] for i in range(test_size)])
test_plaintexts = np.array([test_metadata[i]['plaintext'][subkey] for i in range(test_size)])

# Masks
train_masks = np.array([train_metadata[i]['masks'][subkey - 2] for i in range(train_size)])
test_masks = np.array([test_metadata[i]['masks'][subkey - 2] for i in range(test_size)])

# print("Testing unmasked")
# for i in range(100):
#     p = train_plaintexts[i]
#     k = train_keys[i]
#     z = util.SBOX[p ^ k]
#     print(f"{z} = {y_train_masked[i]}, key={k}")

# Generate key guesses
key_guesses_masked = np.empty((test_size, 256), dtype=np.uint8)
key_guesses_unmasked = np.empty((test_size, 256), dtype=np.uint8)
for trace_num in range(test_size):
    for key_guess in range(256):
        masked = util.SBOX[test_plaintexts[trace_num] ^ key_guess]
        key_guesses_masked[trace_num][key_guess] = masked

        unmasked = masked ^ test_masks[trace_num]
        key_guesses_unmasked[trace_num][key_guess] = unmasked

# Select the paths
path_data_set = path + "/ASCAD_Keys/"

# Traces path
train_traces_path = path_data_set + "/traces/train_traces"
test_traces_path = path_data_set + "/traces/test_traces"

# Train paths
train_model_masked_path = path_data_set + "Value/train_model_masked.csv"
train_model_masked_hw_path = path_data_set + "Value/train_model_hw_masked.csv"
train_model_unmasked_path = path_data_set + "Value/train_model_unmasked.csv"
train_model_unmasked_hw_path = path_data_set + "Value/train_model_hw_unmasked.csv"

# Test paths
test_model_masked_path = path_data_set + "Value/test_model_masked.csv"
test_model_masked_hw_path = path_data_set + "Value/test_model_hw_masked.csv"
test_model_unmasked_path = path_data_set + "Value/test_model_unmasked.csv"
test_model_unmasked_hw_path = path_data_set + "Value/test_model_hw_unmasked.csv"

# Key guesses path
key_guesses_masked_path = path_data_set + "Value/key_guesses_masked.csv"
key_guesses_unmasked_path = path_data_set + "Value/key_guesses_unmasked.csv"

# Plaintexts path
train_plaintexts_path = path_data_set + "Value/train_plaintexts"
test_plaintexts_path = path_data_set + "Value/test_plaintexts"

# Secret key path
secret_key_path = path_data_set + "/secret_key.csv"

# Masks path
test_masks_path = path_ascad + "/Value/test_masks"
train_masks_path = path_ascad + "/Value/train_masks"

pdb.set_trace()

save = False
if save:
    # Save the traces,
    np.save(train_traces_path, x_train_traces)
    np.save(test_traces_path, x_test_traces)

    # Save the key guesses
    np.save(key_guesses_masked_path, key_guesses_masked)
    np.save(key_guesses_unmasked_path, key_guesses_unmasked)

    # Save the train model values
    np.save(train_model_unmasked_path, y_train_unmasked)
    np.save(train_model_unmasked_hw_path, y_train_hw_unmasked)
    np.save(train_model_masked_path, y_train_masked)
    np.save(train_model_masked_hw_path, y_train_hw_masked)

    # Save the test model values
    np.save(test_model_unmasked_path, y_test_unmasked)
    np.save(test_model_unmasked_hw_path, y_test_hw_unmasked)
    np.save(test_model_masked_path, y_test_masked)
    np.save(test_model_masked_hw_path, y_test_hw_masked)

    # Save the plaintexts
    np.save(train_plaintexts_path, train_plaintexts)
    np.save(test_plaintexts_path, test_plaintexts)

    # Save the masks
    np.save(test_masks_path, test_masks)
    np.save(train_masks_path, train_masks)

    # Save the secret key
    with open(secret_key_path, mode="w") as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')
        writer.writerow([test_keys[0]])


normalized = True
if normalized:
    # Scale the data
    scale = StandardScaler()
    x_train_normalized = scale.fit_transform(x_train_traces)
    x_test_normalized = scale.transform(x_test_traces)

    # Set the paths
    path_data_set = path + "/ASCAD_Keys_Normalized/"
    path_x_train_normalized = path_data_set + "/traces/train_traces"
    path_x_test_normalized = path_data_set + "/traces/test_traces"

    # Save the traces
    np.save(path_x_train_normalized, x_train_normalized)
    np.save(path_x_test_normalized, x_test_normalized)
