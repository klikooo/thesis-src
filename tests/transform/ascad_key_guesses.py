import numpy as np
import util

path = "/media/rico/Data/TU/thesis/data/"
path_ascad = path + "ASCAD_Keys/"
path_plaintexts = path_ascad + "/Value/train_plaintexts.npy"
path_masks = path_ascad + "/Value/train_masks.npy"

plaintexts = np.load(path_plaintexts)
masks = np.load(path_masks)

path_train_key_guesses_masked = path_ascad + "/Value/train_key_guesses_masked"
path_train_key_guesses_unmasked = path_ascad + "/Value/train_key_guesses_unmasked"

train_size = 200000
key_guesses_masked = np.empty((train_size, 256), dtype=np.uint8)
key_guesses_unmasked = np.empty((train_size, 256), dtype=np.uint8)
for trace_num in range(train_size):
    for key_guess in range(256):
        masked = util.SBOX[plaintexts[trace_num] ^ key_guess]
        key_guesses_masked[trace_num][key_guess] = masked

        unmasked = masked ^ masks[trace_num]
        key_guesses_unmasked[trace_num][key_guess] = unmasked

np.save(path_train_key_guesses_masked, key_guesses_masked)
np.save(path_train_key_guesses_unmasked, key_guesses_unmasked)



