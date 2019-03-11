import numpy as np
import util


data_set = util.DataSet.AES_HD
data_set_name = str(data_set)

file = '/media/rico/Data/TU/thesis/data/{}/Value/key_guesses_ALL.csv'.format(data_set_name)
key_guesses = np.transpose(
    util.load_csv(file,
                  delimiter=' ',
                  dtype=np.int))

save_file = '/media/rico/Data/TU/thesis/data/{}/Value/key_guesses_ALL_transposed.csv'.format(data_set_name)
np.savetxt(save_file, key_guesses.astype(int), delimiter=" ", fmt="%i")
