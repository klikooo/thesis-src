import util
import numpy as np
import subprocess

data_set = util.DataSet.RANDOM_DELAY
data_loader = util.load_data_set(data_set)

files = []
step = 2000
for i in range(0, 50000, step):
    print(i)
    args = {"raw_traces": True,
            "start": i,
            "size": step,
            "traces_path": "/media/rico/Data/TU/thesis/data/",
            "use_hw": False}

    path_rd = '{}/Random_Delay/traces/'.format(args['traces_path'])

    x_train = util.load_csv('{}/Random_Delay/traces/traces_complete.csv'.format(args['traces_path']),
                            delimiter=' ',
                            start=args.get('start'),
                            size=args.get('size'))
    mean = np.mean(x_train, axis=0)

    noise = np.random.normal(0, 15, 3500 * args['size']).reshape((args['size'], 3500))
    print(np.shape(noise))

    result = x_train + noise

    file = 'traces_noise_{}.csv'.format(i)
    files.append(file)
    util.save_np('{}/{}'.format(path_rd, file), result, f="%i")

command = "cat {} > test.csv".format(" ".join(files))
print(command)
bashCommand = "rm traces_noise_* "
print(bashCommand)
