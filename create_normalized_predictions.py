from models.load_model import load_model
import util
from sklearn.preprocessing import StandardScaler
from test import accuracy, create_key_probabilities_id, create_key_probabilities_hw
import numpy as np
import sys


def load_traces(traces_path, hamming_weight):
    args = util.EmptySpace
    args.load_traces = True
    args.use_hw = hamming_weight
    args.size = 50000
    args.unmask = True
    args.traces_path = traces_path
    args.data_set = util.DataSet.KEYS
    args.raw_traces = True
    args.start = 0
    args.train_size = 0
    args.validation_size = 0
    args.use_noise_data = False
    args.subkey_index = 2
    args.desync = 0
    args.unmask = True
    args.noise_level = 0
    args.attack_size = 50000
    x, y, _, key, key_guesses = util.load_test_data(args)
    return x, y, key, key_guesses


def normalize_traces(traces, num_traces_to_normalize):
    z = [np.random.randint(50000) for _ in range(num_traces_to_normalize)]

    to_normalize = traces[z]

    scale = StandardScaler()
    scale.fit(to_normalize)

    normalized = scale.transform(traces)
    return normalized, z


def load_models(path, runs):
    model_path = path + 'model_r{}_DenseNet.pt'

    models = []
    for run in runs:
        model = load_model("DenseNet", model_path.format(run))
        print(model)
        model.eval()
        model.to(util.device)
        models.append(model)
    return models


def do(path, traces_path, list_num_traces, num_experiments, runs, hw):
    # Load models
    util.w_print("Loading models")
    models = load_models(path, range(runs))

    # Load traces
    util.w_print("Loading traces")
    x, y, key, key_guesses = load_traces(traces_path, hw)

    # Select correct function for calculating key probabilities
    create_key_probabilities_function = create_key_probabilities_hw if hw else create_key_probabilities_id

    # Loop over the list of amount of traces to use
    num_models = len(models)
    for num_traces in list_num_traces:
        util.e_print(f"Using {num_traces} trace(s)")

        # Set up
        sum_accuracy = np.zeros(num_models)
        sum_ge = np.zeros(num_models)

        # Perform num_experiments for these amount of traces
        for i in range(num_experiments):
            # Normalize traces + receive index from which we have normalized from
            normalized_traces, index = normalize_traces(x, num_traces)
            selected_key_guess = key_guesses[index]

            # Dit this for each model
            for j in range(num_models):
                # Calculate predictions
                predictions, acc = accuracy(models[j], normalized_traces, y, None)
                predictions = predictions.cpu().numpy()

                # Calculate the key rank
                key_probabilities = create_key_probabilities_function(selected_key_guess,
                                                                      predictions[index],
                                                                      num_traces)

                summed_probabilities = np.sum(key_probabilities, axis=0)
                sorted_guess = np.argsort(summed_probabilities)
                rank = np.argmax((sorted_guess[::-1] == key))
                print(f"Model {j} guess: {sorted_guess[255]}, key: {key}. rank: {rank}")

                sum_accuracy[j] += acc
                sum_ge[j] += rank

        # Print some stats
        avg_accuracy = sum_accuracy / num_experiments
        avg_ge = sum_ge / num_experiments
        util.e_print(f"Avg acc: {avg_accuracy}")
        util.e_print(f"Avg rank: {avg_ge}")

        # Save the files
        np.save(f'{path}/traces_{num_traces}_avg_accuracy', avg_accuracy)
        np.save(f'{path}/traces_{num_traces}_avg_ge', avg_ge)


def start():
    epochs = 10
    hw = False
    traces_p = '/media/rico/Data/TU/thesis/data/'
    models_p = '/media/rico/Data/TU/thesis/runs3/KEYS/subkey_2/'
    train_size = 1000
    batch_size = 256

    print(sys.argv)
    print(len(sys.argv))

    if len(sys.argv) == 7:
        traces_p = sys.argv[1]
        models_p = sys.argv[2]
        epochs = sys.argv[3]
        train_size = sys.argv[4]
        batch_size = sys.argv[5]
        hw = sys.argv[6] == "True"

    hw_string = "HW" if hw else "ID"

    num_traces = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 50, 100, 200]
    # num_traces = [5000]

    models_p = models_p + f'{hw_string}_SF1_E{epochs}_BZ{batch_size}_LR1.00E-04/train{train_size}/'
    do(models_p, traces_p,
       num_traces,
       num_experiments=100, runs=5, hw=hw)


if __name__ == "__main__":
    start()
