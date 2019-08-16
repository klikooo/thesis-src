from keras.models import load_model
import util
import numpy as np
import pprint


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


def accuracy(predi, y):
    pred = np.argmax(predi, axis=1)

    z = pred == y
    num_correct = np.sum(z)

    print('Correct: {}'.format(num_correct))
    print('Accuracy: {} - {}%'.format(num_correct / len(y), num_correct / len(y) * 100))
    return z


model = load_model('cnn2-ascad-desync0.h5')
printer = pprint.PrettyPrinter(indent=2)
printer.pprint(model.get_config())

hw = False
num_classes = 9 if hw else 256
size = 1000
# For testing for overfit on training data
# x, y, _ = util.load_ascad_keys({
#     "data_set": util.DataSet.ASCAD_KEYS,
#     "traces_path": "/media/rico/Data/TU/thesis/data/",
#     "use_hw": hw,
#     "unmask": False,
#     "size": size,
#     "start": 0
# })
# import sys
# np.set_printoptions(threshold=sys.maxsize)

x, y, key_guesses, key, _ = util.load_ascad_keys_test({
    "data_set": util.DataSet.ASCAD_KEYS,
    "traces_path": "/media/rico/Data/TU/thesis/data/",
    "use_hw": hw,
    "unmask": False,
    "size": size
})

x = x.reshape((x.shape[0], x.shape[1], 1))
predictions = model.predict(x)
acc = accuracy(predictions, y)
ranks = []
for i in range(100):
    permutation = np.random.permutation(size)
    p_key_guesses = key_guesses[permutation]
    p_predictions = predictions[permutation]

    x_rank, y_rank = test_with_key_guess_p(p_key_guesses, p_predictions, hw, key, attack_size=size)
    ranks.append(y_rank)

mean_rank = np.mean(ranks, axis=0)
import matplotlib.pyplot as plt
plt.grid(True)
plt.plot(mean_rank)
plt.show()

import pdb
pdb.set_trace()
