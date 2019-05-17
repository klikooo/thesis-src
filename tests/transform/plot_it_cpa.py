import matplotlib.pyplot as plt
import numpy as np

real_key = 208
begin_path = "/media/rico/Data/TU/thesis/data/Random_Delay_Large"
f = "{}/correlation.npy".format(begin_path)
res = np.load(f)
print(np.shape(res))


def plot(results, num):
    results = np.abs(np.transpose(results))
    if np.sum(results) == 0:
        print("Stopping")
        return False

    plt.figure()
    plt.title("Correlation of first {} traces".format(num))
    # plt.plot(results)
    real = np.array(results)[:, real_key]
    plt.plot(real, label="Real key", marker="*", color='gold')
    plt.legend()
    return True


for i in range(len(res)):
    continue_plot = plot(res[i], (i + 1) * 20000)
    if not continue_plot:
        break

plt.show()
