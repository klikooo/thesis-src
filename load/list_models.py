import os
import numpy as np
import util

data_set = 'Random_Delay'
path = '/media/rico/Data/TU/thesis/runs2/{}/subkey_2/'.format(data_set)


def test_model_works(file, start, threshold=5, spread=10):
    ge = util.load_csv(file, delimiter=' ', dtype=np.float)
    if sum(ge[start-spread:start+spread]) < threshold:
        return True
    return False


for setting_dir in os.listdir(path):
    if setting_dir.startswith('ID_SF1'):
        setting_path = '{}/{}'.format(path, setting_dir)
        for train_dir in os.listdir(setting_path):
            train_path = '{}/{}'.format(setting_path, train_dir)
            print(train_path)
            files = [file for file in os.listdir(train_path) if file.endswith(".exp")]
            works = []
            for file in files:
                file_path = '{}/{}'.format(train_path, file)
                model_name = file[9:]

                if model_name not in works and test_model_works(file_path, 150):
                    print('Works: {}'.format(model_name))
                    works.append(model_name)




