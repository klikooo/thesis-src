import util
from util import load_ascad, HW, SBOX
import numpy as np

path = '/media/rico/Data/TU/thesis'
desync = 100
sub_key_index = 2
unmask = True
use_hw = False

s_index = 2

trace_file = '{}/data/ASCAD/ASCAD_{}_desync{}.h5'.format(path, sub_key_index, desync)

(_, _), (x_attack, y_attack), (metadata_profiling, metadata_attack) = \
    load_ascad(trace_file, load_metadata=True)

# Perform some operations that are desired
if unmask:
    y_attack = np.array([y_attack[i] ^ metadata_attack[i]['masks'][0] for i in range(len(y_attack))])
if use_hw:
    y_attack = np.array([HW[val] for val in y_attack])

key_guesses = []
for i in range(len(y_attack)):
    plaintext = metadata_attack[i]['plaintext'][s_index]
    trace_key_guess = []
    for key_guess in range(256):
        res = (SBOX[plaintext ^ key_guess] ^ metadata_attack[i]['masks'][0]).astype(np.int)
        trace_key_guess.append(res)

    key_guesses.append(trace_key_guess)
    # print(trace_key_guess)
    # print(len(trace_key_guess))

key_guesses = np.array(key_guesses)
np.savetxt('/media/rico/Data/TU/thesis/data/ASCAD/key_guesses.csv',
           key_guesses, delimiter=' ', fmt='%i')

key_guesses = util.load_csv('/media/rico/Data/TU/thesis/data/ASCAD/key_guesses.csv',
                  delimiter=' ',
                  dtype=np.int,
                  start=0,
                  size=100)


index = 0
z = y_attack[index]
k = metadata_attack[index]['key'][s_index]
p = metadata_attack[index]['plaintext'][s_index]
print("res sbox: {}".format(SBOX[p^k]))
print("y: {}\nk: {}".format(z, k))
print("z*: {}".format(key_guesses[index][k]))


print('')
# print(key_guesses[0])
