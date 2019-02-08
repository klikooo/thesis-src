
import os, os.path
target = 40


DIR = '/media/rico/Data/TU/thesis/runs2/DPAv4/subkey_2'

for d in os.listdir(DIR):
    train_dirs = os.listdir(os.path.join(DIR, d))
    if len(train_dirs) == 0:
        print('EMPTY {}/{}'.format(DIR, d))
    for d2 in train_dirs:
        if len(d2) == 0:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")

        z = os.listdir(os.path.join(os.path.join(DIR,d),d2))
        if len(z) == 0:
            print('EMPTY TRAIN')

        print('{} {}'.format(os.path.join(os.path.join(DIR, d), d2), len(z)))
        if len(z) != target:
            print('DID NOT MET TARGET')
        # print(os.listdir(z))
    # exit()
# path joining version for other paths
# print(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
# ID_SF3_E80_BZ1000_LR1.00E-02
#ID_SF3_E80_BZ1000_LR1.00E-03