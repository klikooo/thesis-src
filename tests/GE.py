import numpy as np
import os


# translate probabilities of label classes to keys guesses
# HW/HD: 9 classes => 256 key guesses
# value model: 256 classes => 256 key guesses
# input: label probabilities, key guesses (depending on the crypto alg/operation attacked, externally computed)
# output: probabilities of each key

def calculate_key_proba(label_proba, key_guess):
    key_proba = np.zeros(256)

    i = 0
    for key in np.nditer(key_guess):
        key_proba[i] = label_proba[int(key)]
        i = i + 1
    return key_proba


# computing guessing entropy, and success rate
# SR: is the most probable key == secret key
# GE: ranking position of the correct key
# input: key probabilities, secret key
# output: guessing entropy, success rate (for one sample)

def GE_SR(key_proba, secret_key):
    ranking = key_proba.argsort()
    SR = 0
    if ranking[-1] == secret_key:
        SR = 1

    tmp = (ranking == secret_key).nonzero()
    GE = 256 - tmp[0]

    return (GE, SR)


# computes GE/SR for a given number of traces, over a number of experiments
# input: number of traces, number of experiments, path where the key guesses and secret key are stored, path where the label probabilities are stored, filename of the label probabilities

def computing(number_traces, number_exp, path_to_guesses, path_to_files, method):
    keyguesses = np.loadtxt(path_to_guesses + 'key_guesses_test.csv')
    label_proba = np.loadtxt(path_to_files + method, delimiter=',')
    label_proba_log = np.zeros(label_proba.shape)
    for index, x in np.ndenumerate(label_proba):
        if label_proba[index] > 0.0:
            label_proba_log[index] = np.log(x)

    secret_key = np.loadtxt(path_to_guesses + 'secret_key.csv')

    SR = np.zeros(number_traces)
    GE = np.zeros(number_traces)

    for exp in range(number_exp):
        indexes = np.arange(number_traces)

        np.random.shuffle(indexes)
        key_proba_total = np.zeros(256)

        for i in range(number_traces):
            ind = indexes[i]
            key_proba_total = key_proba_total + calculate_key_proba(label_proba_log[ind, :], keyguesses[:, ind])
            ge, sr = GE_SR(key_proba_total, secret_key)

            SR[i] = SR[i] + sr
            GE[i] = GE[i] + ge

    SR = SR / float(number_exp)
    GE = GE / float(number_exp)
    return SR, GE


def computing_threshold(keyguesses, secret_key, label_proba, log, number_exp):
    if log:
        label_proba_log = label_proba
    else:
        label_proba_log = np.zeros(label_proba.shape)
        for index, x in np.ndenumerate(label_proba):
            if label_proba[index] > 0.0:
                label_proba_log[index] = np.log(x)

    number_traces = label_proba.shape[0]
    SR = np.zeros(number_traces)
    GE = np.zeros(number_traces)

    for exp in range(number_exp):
        indexes = np.arange(number_traces)

        np.random.shuffle(indexes)
        key_proba_total = np.zeros(256)

        for i in range(number_traces):
            ind = indexes[i]
            key_proba_total = key_proba_total + calculate_key_proba(label_proba_log[ind, :], keyguesses[:, ind])
            ge, sr = GE_SR(key_proba_total, secret_key)

            SR[i] = SR[i] + sr
            GE[i] = GE[i] + ge

    SR = SR / float(number_exp)
    GE = GE / float(number_exp)
    return SR, GE


def get_guessing_GE_level(keyguesses, secret_key, predictions, log):
    number_exp = 20
    SR, GE = computing_threshold(keyguesses, secret_key, predictions, log, number_exp)
    thres_GE = np.argmax(GE < 10)
    thres_SR = np.argmax(SR > 0.9)
    return thres_GE, thres_SR


def main():
    # my code here

    number_of_traces = 100  # DPAv4: 100, all others: 25000
    number_of_exp = 100  # 50 should also be sufficient in case speed up is needed
    path_to_guesses = ''  # path to key guesses and secret key
    path_to_files = ''  # path where label probabilities are stored (delimiter=',', or to be changed in function above)
    method = 'bla.csv'  # filename of label probabilities

    SR, GE = computing(number_of_traces, number_of_exp, path_to_guesses, path_to_files, method)

    # saving
    steps = range(number_of_traces)
    np.savetxt(path_to_files + method[:-4] + '_SR.csv', [steps, SR], delimiter=',')
    np.savetxt(path_to_files + method[:-4] + '_GE.csv', [steps, GE], delimiter=',')


if __name__ == "__main__":
    main()

