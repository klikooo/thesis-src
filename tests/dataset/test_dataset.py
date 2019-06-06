import numpy as np

path = "/media/rico/Data/TU/thesis/data/Simulated_Mask/"
traces_path = path + "traces/"
traces_file = traces_path + "traces.npy"

value_path = path + "Value/"
key_guesses_file = value_path + "key_guesses_ALL_transposed.csv.npy"
model_file = value_path + "model.csv.npy"


traces = np.load(traces_file)
print(traces.shape)

key_guesses = np.load(key_guesses_file)
print(key_guesses.shape)

model = np.load(model_file)
print(model.shape)


