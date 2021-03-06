from util import load_ascad, HW, SBOX
import numpy as np

path = '/media/rico/Data/TU/thesis'
desync = 0
sub_key_index = 2
unmask = True
use_hw = False

s_index = 2

trace_file = '{}/data/ASCAD/ASCAD_{}_desync{}.h5'.format(path, sub_key_index, desync)

(_, _), (_, _), (_, metadata_attack) = \
    load_ascad(trace_file, load_metadata=True)

print("Loaded traces")
# Perform some operations that are desired
# if unmask:
#     y_attack = np.array([y_attack[i] ^ metadata_attack[i]['masks'][0] for i in range(len(y_attack))])
# if use_hw:
#     y_attack = np.array([HW[val] for val in y_attack])

key_guesses = np.zeros((len(metadata_attack), 256), dtype=int)
print("shape key guesses: {}".format(np.shape(key_guesses)))
print("Len of metadata attack: {}".format(len(metadata_attack)))
for i in range(len(metadata_attack)):
    plaintext = metadata_attack[i]['plaintext'][s_index]
    for key_guess in range(256):
        # res = (SBOX[plaintext ^ key_guess] ^ metadata_attack[i]['masks'][0]).astype(np.int)
        key_guesses[i][key_guess] = SBOX[plaintext ^ key_guess] ^ metadata_attack[i]['masks'][0]
        # trace_key_guess.append(res)
    # print(i)
    # key_guesses.append(trace_key_guess)
    # print(trace_key_guess)
    # print(len(trace_key_guess))

print("Saving key guesses")
# key_guesses = np.array(key_guesses)
np.save('/media/rico/Data/TU/thesis/data/ASCAD/key_guesses_unmasked',
        key_guesses)
print("Saved.\nStarting with masked key guesses")

key_guesses = np.zeros((len(metadata_attack), 256), dtype=int)
for i in range(len(metadata_attack)):
    plaintext = metadata_attack[i]['plaintext'][s_index]
    for key_guess in range(256):
        # res = (SBOX[plaintext ^ key_guess] ^ metadata_attack[i]['masks'][0]).astype(np.int)
        key_guesses[i][key_guess] = SBOX[plaintext ^ key_guess]
np.save('/media/rico/Data/TU/thesis/data/ASCAD/key_guesses_masked',
        key_guesses)


# ks = util.load_csv('/media/rico/Data/TU/thesis/data/ASCAD/key_guesses.csv',
#                    delimiter=' ',
#                    dtype=np.int,
#                    start=0,
#                    size=100)
#
# index = 0
# z = y_attack[index]
# k = metadata_attack[index]['key'][s_index]
# p = metadata_attack[index]['plaintext'][s_index]
# print("res sbox: {}".format(SBOX[p ^ k]))
# print("y: {}\nk: {}".format(z, k))
# print("z*: {}".format(ks[index][k]))

