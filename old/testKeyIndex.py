from util import load_csv, SBOX
import numpy as np


size = 10
plaintext = '/home/rico/Downloads/RD_traces/traceplaintext.csv'
plains = load_csv(plaintext,
                  delimiter=' ',
                  size=size,
                  dtype=np.int)

key_guess_f = '/media/rico/Data/TU/thesis/data/Random_Delay/Value/key_guesses_ALL_transposed.csv'
model_f = '/media/rico/Data/TU/thesis/data/Random_Delay/Value/model.csv'
model = load_csv(model_f,
                 delimiter=' ',
                 size=size,
                 dtype=np.int)
key = 43
# print(model)
for i in range(size):
    index = 0
    for p in plains[i]:
        s = SBOX[p ^ key]
        if s == model[i]:
            print('Subkey {} at trace {}'.format(index, i))
        index += 1
