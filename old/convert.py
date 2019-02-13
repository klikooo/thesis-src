import scipy.io
import numpy as np

data = scipy.io.loadmat("/home/rico/Downloads/ctraces_fm16x4_2.mat")

for i in data:
    if '__' not in i and 'readme' not in i:
        if i in 'plaintext':
            d = data[i].astype(int)
            np.savetxt(("/home/rico/Downloads/RD_traces/trace" + i + ".csv"), d, delimiter=' ', fmt="%i")
        # else:
        #     np.savetxt(("/home/rico/Downloads/RD_traces/trace" + i + ".csv"), data[i], delimiter=',')
