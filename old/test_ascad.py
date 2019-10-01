from keras.models import load_model
import util
import numpy as np
import pprint
from test import create_key_probabilities, test_with_key_probabilities



def accuracy(predi, y):
    pred = np.argmax(predi, axis=1)

    z = pred == y
    num_correct = np.sum(z)

    print('Correct: {}'.format(num_correct))
    print('Accuracy: {} - {}%'.format(num_correct / len(y), num_correct / len(y) * 100))
    return z


model = load_model('test_model_cnn')
printer = pprint.PrettyPrinter(indent=2)
printer.pprint(model.get_config())

hw = False
num_classes = 9 if hw else 256
size = 10000

x, y, _, key, key_guesses = util.load_ascad_test_traces({
    "data_set": util.DataSet.ASCAD,
    "traces_path": "/media/rico/Data/TU/thesis/data/",
    "use_hw": hw,
    "unmask": False,
    "size": size,
    "desync": 0,
    "sub_key_index": 2,
})

x = x.reshape((x.shape[0], x.shape[1], 1))
predictions = model.predict(x)
acc = accuracy(predictions, y)
ranks = []

print("Key guesses")
print(np.shape(key_guesses))

print("Predictions")
print(predictions)
print(np.shape(predictions))


key_probabilities = create_key_probabilities(key_guesses, predictions, size, hw)
print("Key proba")
print(key_probabilities)
print(np.shape(key_probabilities))
for i in range(100):
    permutation = np.random.permutation(size)
    p_key_guesses = key_guesses[permutation]
    p_predictions = predictions[permutation]

    key_probabilities_shuffled = util.shuffle_permutation(permutation, key_probabilities)

    x_rank, y_rank, k_guess = test_with_key_probabilities(key_probabilities_shuffled, key)
    ranks.append(y_rank)

mean_rank = np.mean(ranks, axis=0)
import matplotlib.pyplot as plt
plt.grid(True)
plt.plot(mean_rank)
plt.show()

import pdb
pdb.set_trace()
