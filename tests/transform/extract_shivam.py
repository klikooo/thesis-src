import numpy as np
import csv
import util


path = "/home/rico/Downloads/shivam/"

traces_filename = "trace_dataset3.csv"
key_filename = "key_dataset3.csv"
input_filename = "input_dataset3.csv"

num_traces = 50000


def read_csv(filepath):
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in reader:
            data.append([int(x) for x in row])
    return np.array(data)


traces = read_csv(f'{path}{traces_filename}')
keys = np.reshape(np.transpose(read_csv(f'{path}{key_filename}')), (num_traces,))
plaintexts = np.reshape(np.transpose(read_csv(f'{path}{input_filename}')), (num_traces,))


all_key_guesses = []
for trace_index in range(num_traces):
    plain = plaintexts[trace_index]
    guess = [util.SBOX[plain ^ key_guess] for key_guess in range(256)]
    all_key_guesses.append(guess)

key_guesses = np.array(all_key_guesses)

save_path = "/media/rico/Data/TU/thesis/data/KEYS"


