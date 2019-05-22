import matplotlib.pyplot as plt
import numpy as np
import os

real_key = 208
begin_path = "/media/rico/Data/TU/thesis/data/Random_Delay_Large"


def plot(num):
    # results = np.abs(np.transpose(results))
    file = "{}/correlation_{}.npy".format(begin_path, num)
    if not os.path.isfile(file):
        return False
    result = np.load(file)
    result = np.abs(np.transpose(result))

    plt.figure()
    plt.title("Correlation of first {} traces".format(num))
    plt.plot(result)
    real = np.array(result)[:, real_key]
    plt.plot(real, label="Real key", marker="*", color='gold', markevery=0.1)
    plt.legend()
    return True


for i in range(200):
    continue_plotting = plot((i + 1) * 20000)
    if not continue_plotting:
        break

plt.show()
