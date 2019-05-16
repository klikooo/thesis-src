import util
import numpy as np

path = "/media/rico/Data/TU/thesis/data/Random_Delay/"
key_guesses_path = path + "Value/"
traces_path = path + "traces/"

traces_filename = traces_path + "traces_complete.csv"
key_guesses_filename = key_guesses_path + "key_guesses_ALL_transposed.csv"
model_filename = key_guesses_path + "model.csv"

# def traces():
#     total_size = 50000
#     size = 1000
#     num_steps = int(total_size / size)
#     num_features = 3500
#
#     all_traces = np.zeros((total_size, num_features))
#     for i in range(num_steps):
#         start = i * size
#         print("Start = {}".format(start))
#         x = util.load_csv(traces_filename,
#                           delimiter=' ',
#                           start=start,
#                           size=size)
#         all_traces[start:start+size] = x
#
#     print(all_traces.shape)
#     np.save(new_traces_filename, all_traces)


def to_npy(step_size, num_features, filename):
    total_size = 50000
    num_steps = int(total_size / step_size)

    all = np.zeros((total_size, num_features))
    for i in range(num_steps):
        start = i * step_size
        print("Start = {}".format(start))
        x = util.load_csv(filename,
                          delimiter=' ',
                          start=start,
                          size=step_size)
        if x.shape == (step_size,):
            x = x.reshape((step_size, 1))
        print(x.shape)

        all[start:start + step_size] = x

    print(all.shape)
    np.save(filename, all)


# to_npy(1000, 3500, traces_filename)
# to_npy(1000, 256, key_guesses_filename)
to_npy(1000, 1, model_filename)
