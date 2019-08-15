from keras.models import load_model
import util
import numpy as np


def test_with_key_guess_p(key_guesses, predictions, use_hw, real_key,
                          attack_size=10000,
                          ):
    ranks = np.zeros(attack_size)
    probabilities = np.zeros(256)
    if not use_hw:
        for trace_num in range(attack_size):
            for key_guess in range(256):
                sbox_out = key_guesses[trace_num][key_guess]
                if predictions[trace_num][sbox_out] > 0.0:
                    probabilities[key_guess] += np.log(predictions[trace_num][sbox_out])

            res = np.argmax(np.argsort(probabilities)[::-1] == real_key)
            ranks[trace_num] = res
    else:
        for trace_num in range(attack_size):
            for key_guess in range(256):
                sbox_out = key_guesses[trace_num][key_guess]
                probabilities[key_guess] += predictions[trace_num][util.HW[sbox_out]]
            res = np.argmax(np.argsort(probabilities)[::-1] == real_key)
            ranks[trace_num] = res

    print('Key guess: {}'.format(np.argmax(probabilities)))

    return np.array(range(1, attack_size + 1)), ranks


model = load_model('cnn2-ascad-desync0.h5')

from keras.utils import plot_model
plot_model(model, to_file='model.png')
config = model.get_config()
import pprint
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(config)
# print(config)
import pdb
pdb.set_trace()
exit()


hw = False
num_classes = 9 if hw else 256
attack_size = 1000
x, y, key_guesses, key, _ = util.load_ascad_keys_test({
    "data_set": util.DataSet.ASCAD_KEYS,
    "traces_path": "/media/rico/Data/TU/thesis/data/",
    "use_hw": hw,
    "unmask": True,
    "size": attack_size
})

x = x.reshape((x.shape[0], x.shape[1], 1))
predictions = model.predict(x)

x_rank, y_rank = test_with_key_guess_p(key_guesses, predictions, hw, key, attack_size=attack_size)

print(y_rank)

import pdb
pdb.set_trace()
