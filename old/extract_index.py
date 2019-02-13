import numpy as np

from util import load_csv

# size = 10
plaintext = '/home/rico/Downloads/RD_traces/traceplaintext.csv'
plains = load_csv(plaintext,
                  delimiter=' ',
                  # size=size,
                  dtype=np.int)

subkey_index = 0
data = plains[:, subkey_index].astype(int)
np.savetxt("/home/rico/Downloads/RD_traces/plain_{}.csv".format(subkey_index),
           data,
           delimiter=' ',
           fmt="%i")
print(np.shape(data))