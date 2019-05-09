import util
import numpy as np


def generate_key_guesses(plain):
    kguesses = []
    for kguess in range(0, 256):
        sbox_out = util.SBOX[plain ^ kguess]
        kguesses.append(sbox_out)
    return np.array(kguesses)


start = 0
size = 50000
traces_path = '/media/rico/Data/TU/thesis/data/'

args = {
    "use_hw": "Value",
    "raw_traces": True,
    "start": start,
    "size": size,
    "traces_path": traces_path,
    "domain_knowledge": False
}

y_train = util.load_csv('{}/Random_Delay/{}/model.csv'.format(args['traces_path'], args['use_hw']),
                        delimiter=' ',
                        dtype=np.long,
                        start=args.get('start'),
                        size=args.get('size'))
print(np.shape(y_train))

random_key_vector = np.random.randint(0, 256, size=size)


plaintexts = []
key_guesses = []
for i in range(size):
    z = y_train[i]
    k = random_key_vector[i]
    p = util.SBOX_INV[z] ^ k

    plaintexts.append(p)
    key_guesses.append(generate_key_guesses(p))

    # a = generate_key_guesses(p)
    # print("guess: {}".format(a[k]))

    # print("p={}, k={}, SBOX={}, z={}".format(p, k, util.SBOX[p ^ k], z))

print("Shape plain: {}".format(np.shape(plaintexts)))
print("Shape key guesses: {}".format(np.shape(key_guesses)))

data_set_name = "Random_Delay_DK"
util.save_np("{}/{}/Value/plaintexts.csv".format(traces_path, data_set_name), plaintexts)
util.save_np("{}/{}/Value/key_guesses.csv".format(traces_path, data_set_name), key_guesses)
