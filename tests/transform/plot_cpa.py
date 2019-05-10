import matplotlib.pyplot as plt
import numpy as np
import util

real_key = 208
begin_path = "/media/rico/Data/TU/thesis/data/"
f = "{}/cpa_plot.npy".format(begin_path)
res = np.array(util.load_csv(f, delimiter=' ', dtype=np.float))
print(np.shape(res))
print(res[0])

def plot(results):
    plt.figure()
    plt.plot(results)
    real = np.array(results)[:, real_key]
    plt.plot(real, label="Real key", marker="*", color='gold')
    plt.legend()
    plt.show()


plot(res)