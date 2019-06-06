import util
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


args = {
    "start": 0,
    "size": 2000000,
    "traces_path": '/media/rico/Data/TU/thesis/data/',
    "use_hw": False,
    "raw_traces": True
}


def load_random_delay_large(args):
    print(args)
    traces_step = 20000
    total_steps = np.math.ceil((args['start'] + args['size']) / traces_step)
    y_train = np.zeros((args['size']))
    start_step = int(args['start'] / traces_step)

    index_start = 0
    for step in range(start_step, total_steps):
        y_file = '{}/Random_Delay_Large/Value/model_{}.csv.npy'.format(args['traces_path'], traces_step * (step + 1))
        y = np.load(y_file)

        # Begin step
        if step == start_step:
            if step == total_steps-1:
                y_train[0:args['size']] = y[args['start']:args['start']+args['size']]
            # More steps to come
            else:
                y_train[0:traces_step-args['start']] = y[args['start']:traces_step]
                index_start = traces_step-args['start']
        # Last step
        elif step == total_steps-1:
            y_train[index_start:args['size']] = y[0:args['size']-index_start]
        # More steps to come
        else:
            y_train[index_start:index_start+traces_step] = y[0:traces_step]
            index_start += traces_step
    print(y_train.shape)
    return y_train


def load_random_delay_normal(args):
    return np.load("{}/Random_Delay_Normalized/Value/model.csv.npy".format(args['traces_path']))


y = load_random_delay_large(args)
sns.distplot(y, 256)

y = load_random_delay_normal(args)
sns.distplot(y, 256)

plt.show()
