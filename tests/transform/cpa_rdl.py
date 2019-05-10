import threading

import util
import numpy as np

# import matplotlib.pyplot as plt


# begin_path = "/media/rico/Data/TU/thesis/data/"
begin_path = "/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/student-datasets/"

data_set = "03_19_fpga_aes_ches11_desynch"
path = "{}/RD2/{}/matlab_format".format(begin_path, data_set)
value_path = "{}/Random_Delay_Large/Value/".format(begin_path)
traces_path = "{}/Random_Delay_Large/traces/".format(begin_path)
key_path = "{}/Random_Delay_Large/".format(begin_path)
key_guesses_filename = "key_guesses_ALL_transposed.csv"
traces_filename = "traces.csv"
model_filename = "model.csv"

last_r_key = [0xd0, 0x14, 0xf9, 0xa8, 0xc9, 0xee, 0x25, 0x89, 0xa1, 0x3f, 0x0c, 0xc8, 0xb6, 0x63, 0x0c, 0xa6]
master_key = bytes([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c])


num_traces = 50000
traces = np.array(util.load_csv("{}/{}".format(traces_path, traces_filename), start=0, size=num_traces, delimiter=' ',
                                dtype=np.int))
key_guesses = np.array(util.load_csv("{}/{}".format(value_path, key_guesses_filename), start=0, size=num_traces,
                                     delimiter=' ', dtype=np.int))
real_key = 208
print(traces[0])

print("Shape key guesses: {}".format(np.shape(key_guesses)))
print("Shape traces: {}".format(np.shape(traces)))

num_features = 6250
num_threads = 5
plot_real_key = [[], [], [], [], []]


def threaded(step, prk):
    split = int(num_features / num_threads)
    for trace_point in range(split * step, split * (step+1)):
        probabilities = [0] * 256
        for kguess in range(0, 256):
            kguess_vals = key_guesses[:, kguess]
            traces_points = traces[:, trace_point]
            corr = np.corrcoef(traces_points, kguess_vals)[1][0]
            # print("For kguess {}: {}".format(kguess, corr))
            probabilities[kguess] = abs(corr)
        prk[step].append(probabilities)


# def plot(results):
#     plt.figure()
#     plt.plot(results)
#     real = np.array(results)[:, real_key]
#     plt.plot(real, label="Real key", marker="*", color='gold')
#     plt.legend()
#     plt.show()


threads = []
for arg in range(num_threads):
    t = threading.Thread(target=threaded, args=(arg, plot_real_key))
    threads.append(t)
    t.start()
# Wait for them to finish
for p in threads:
    p.join()
    print('Joined process')

total = plot_real_key[0] + plot_real_key[1] + plot_real_key[2] + plot_real_key[3] + plot_real_key[4]
# plot(total)


util.save_np("{}/cpa_plot.npy".format(begin_path), total, f="%f")
