import util
import numpy as np

path = "/media/rico/Data/TU/thesis/data/Random_Delay/"
key_guesses_path = path + "Value/"
traces_path = path + "traces/"

traces_filename = traces_path + "traces_complete.csv"
key_guesses_filename = key_guesses_path + "key_guesses_ALL_transposed.csv"
model_filename = key_guesses_path + "model.csv"


def to_npy(total_size, step_size, num_features, filename):
    num_steps = int(total_size / step_size)
    print("number steps: {}".format(num_steps))

    all_samples = np.zeros((total_size, num_features))

    for i in range(num_steps):
        start = i * step_size
        print("Step: {}. Start = {}".format(i, start))
        x = util.load_csv(filename,
                          delimiter=' ',
                          start=start,
                          size=step_size)
        if x.shape == (step_size,):
            x = x.reshape((step_size, 1))
        print("Shape of data {}".format(x.shape))

        all_samples[start:start + step_size] = x

    print(all_samples.shape)
    np.save(filename, all_samples)


rd_t_size = 50000
rd_step = 1000
# to_npy(rd_t_size, rd_step, 3500, traces_filename)
# to_npy(rd_t_size, rd_step, 256, key_guesses_filename)
# to_npy(rd_t_size, rd_step, 1, model_filename)


path = "/media/rico/Data/TU/thesis/data/DPAv4/"
key_guesses_path = path + "Value/"
traces_path = path + "traces/"

dpa_traces_filename = traces_path + "traces_complete.csv"
dpa_key_guesses_filename = key_guesses_path + "key_guesses_ALL_transposed.csv"
dpa_model_filename = key_guesses_path + "model.csv"


dpa_t_size = 100000
dpa_step = 1000
# to_npy(dpa_t_size, dpa_step, 3000, dpa_traces_filename)
# to_npy(dpa_t_size, dpa_step, 256, dpa_key_guesses_filename)
# to_npy(dpa_t_size, dpa_step, 1, dpa_model_filename)
