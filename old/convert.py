import scipy.io
import numpy as np

data = scipy.io.loadmat("/home/rico/Downloads/DPA_contestv4_2/k01/DPACV42_005000.trc")

for i in data:
    if '__' not in i and 'readme' not in i:
        if i in 'plaintext':
            d = data[i].astype(int)
            np.savetxt(("/home/rico/Downloads/DPA/trace" + i + ".csv"), d, delimiter=' ', fmt="%i")
        else:
            np.savetxt(("/home/rico/Downloads/DPA/trace" + i + ".csv"), data[i], delimiter=',')
