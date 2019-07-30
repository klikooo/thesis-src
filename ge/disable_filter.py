import math
import pdb

import numpy as np
from test import test_with_key_guess_p, accuracy
from util import shuffle_permutation, save_np, generate_folder_name
from util_classes import get_save_name
from models.load_model import load_model
from multiprocessing import Process
from load.load_threaded import load_data
import torch
import util
import os


def disable_filter(model):
    conv_layer = model.cnn[0]
    weights = conv_layer.weight
    biases = conv_layer.bias
    print(model)

    old_weight = weights[0].clone()
    old_bias = biases[0].clone()

    zero_weight = torch.zeros(weights.size()[1], weights.size()[2])
    zero_bias = torch.zeros(1)

    predictions = []
    correct_indices = []
    sum_indices = [0] * len(y_attack)

    for i in range(weights.size()[0]):
        if i != 0:
            weights[i - 1] = old_weight
            biases[i - 1] = old_bias

        old_weight = weights[i].clone()
        old_bias = biases[i].clone()
        weights[i] = zero_weight
        biases[i] = zero_bias

        print(f"Index {i}")
        prediction = accuracy(model, x_attack, y_attack, dk_plain)

        print(prediction)
        max_values, indices = prediction.max(1)
        res = indices.long() == torch.from_numpy(y_attack.reshape(len(y_attack))).long().to(util.device)

        sum_indices += res.cpu().numpy()
        correct_index = res.nonzero().view(-1)

        correct_indices.append(correct_index)
        predictions.append(prediction.cpu().numpy())

    return predictions, correct_indices, sum_indices


def disable_filter2(model):
    conv_layer = model.cnn[4]
    weights = conv_layer.weight
    biases = conv_layer.bias
    print(model)

    old_weight = weights[0].clone()
    old_bias = biases[0].clone()

    zero_weight = torch.zeros(weights.size()[1], weights.size()[2])
    zero_bias = torch.zeros(1)

    predictions = []

    for i in range(weights.size()[0]):
        if i != 0:
            weights[i - 1] = old_weight
            biases[i - 1] = old_bias

        old_weight = weights[i].clone()
        old_bias = biases[i].clone()
        weights[i] = zero_weight
        biases[i] = zero_bias

        print(f"Index {i}")
        prediction = accuracy(model, x_attack, y_attack, dk_plain)
        predictions.append(prediction.cpu().numpy())

    return predictions


def disable_filter3(model):
    model.to(util.device)
    conv_layer = model.cnn[0]
    print(model)

    weights = conv_layer.weight.clone()
    biases = conv_layer.bias.clone()

    conv_layer.weight = torch.nn.Parameter(
        torch.zeros(weights.size()[0], weights.size()[1], weights.size()[2]).float().to(util.device))
    conv_layer.bias = torch.nn.Parameter(
        torch.zeros(weights.size()[0]).float().to(util.device))
    conv_layer.weight.to(util.device)
    conv_layer.bias.to(util.device)

    predictions = []
    correct_indices = []
    sum_indices = [0] * len(y_attack)

    for i in range(weights.size()[0]):
        if i != 0:
            conv_layer.weight[i - 1] = torch.zeros(weights.size()[1], weights.size()[2])
            conv_layer.bias[i - 1] = torch.zeros(weights.size()[1])

        conv_layer.weight[i] = weights[i]
        conv_layer.bias[i] = biases[i]

        print(f"Index {i}")
        prediction = accuracy(model, x_attack, y_attack, dk_plain)

        max_values, indices = prediction.max(1)
        res = indices.long() == torch.from_numpy(y_attack.reshape(len(y_attack))).long().to(util.device)

        sum_indices += res.cpu().numpy()
        correct_index = res.nonzero().view(-1)

        correct_indices.append(correct_index)
        predictions.append(prediction.cpu().numpy())

    return predictions, correct_indices, sum_indices


def disable_filter4(model):
    model.to(util.device)
    conv_layer = model.cnn[0]
    conv_layer2 = model.cnn[4]
    bn_layer = model.cnn[3]
    print(model)

    weights = conv_layer.weight.clone()
    biases = conv_layer.bias.clone()

    weights_cnn_layer2 = conv_layer2.weight.clone()
    biases_cnn_layer2 = conv_layer2.bias.clone()

    biases_bn_layer = bn_layer.bias

    conv_layer.weight = torch.nn.Parameter(
        torch.zeros(weights.size()[0], weights.size()[1], weights.size()[2]).float().to(util.device))
    conv_layer.bias = torch.nn.Parameter(
        torch.zeros(weights.size()[0]).float().to(util.device))
    conv_layer.weight.to(util.device)
    conv_layer.bias.to(util.device)

    conv_layer2.weight = torch.nn.Parameter(
        torch.zeros(weights_cnn_layer2.size()[0], weights_cnn_layer2.size()[1],
                    weights_cnn_layer2.size()[2]).float().to(util.device))
    conv_layer2.bias = torch.nn.Parameter(
        torch.zeros(biases_cnn_layer2.size()[0]).float().to(util.device))
    conv_layer2.weight.to(util.device)
    conv_layer2.bias.to(util.device)

    # biases_cnn_layer2.bias = torch.nn.Parameter(
    #     torch.zeros(biases_cnn_layer2.size()[0]).float().to(util.device))
    # biases_cnn_layer2.bias.to(util.device)

    predictions = []
    correct_indices = []
    sum_indices = [0] * len(y_attack)

    for i in range(weights.size()[0]):
        if i != 0:
            conv_layer.weight[i - 1] = torch.zeros(weights.size()[1], weights.size()[2])
            conv_layer2.weight[i - 1] = torch.zeros(weights_cnn_layer2.size()[1], weights_cnn_layer2.size()[2])
            conv_layer.bias[i - 1] = torch.zeros(weights.size()[1])
            conv_layer2.bias[i - 1] = torch.zeros(1)
            # bn_layer.bias[i - 1] = torch.zeros(1)

        conv_layer.weight[i] = weights[i]
        conv_layer2.weight[i] = weights_cnn_layer2[i]
        conv_layer.bias[i] = biases[i]
        conv_layer2.bias[i] = biases_cnn_layer2[i]
        # bn_layer.bias[i] = biases_bn_layer[i]

        print(f"Index {i}")
        prediction = accuracy(model, x_attack, y_attack, dk_plain)

        max_values, indices = prediction.max(1)
        res = indices.long() == torch.from_numpy(y_attack.reshape(len(y_attack))).long().to(util.device)

        sum_indices += res.cpu().numpy()
        correct_index = res.nonzero().view(-1)

        correct_indices.append(correct_index)
        predictions.append(prediction.cpu().numpy())

    return predictions, correct_indices, sum_indices


def get_ranks(args, network_name, model_params, edit_model=disable_filter):
    folder = "{}/{}/".format(args.models_path, generate_folder_name(args))

    # Calculate the predictions before hand
    # TODO: for multiple runs
    model_path = '{}/model_r{}_{}.pt'.format(
        folder,
        args.run,
        get_save_name(network_name, model_params))
    print('path={}'.format(model_path))

    if not os.path.exists(f"{model_path}.predictions1.npy"):

        # Load the data and make it global
        global x_attack, y_attack, dk_plain, key_guesses
        x_attack, y_attack, key_guesses, real_key, dk_plain = load_data(args, network_name)
        model = load_model(network_name, model_path)
        model.eval()
        model.to(args.device)

        predictions, correct_indices, sum_indices = edit_model(model)

        np_predictions = np.array(predictions)
        np_correct_indices = np.array(correct_indices)
        np_sum_indices = np.array(sum_indices)
        np.save(f"{model_path}.predictions1", np_predictions)
        np.save(f"{model_path}.correct_indices", np_correct_indices)
        np.save(f"{model_path}.sum_indices", np_sum_indices)
        print(sum_indices)
    else:
        predictions = np.load(f"{model_path}.predictions1.npy")
        real_key = util.load_csv('{}/{}/secret_key.csv'.format(args.traces_path, str(load_args.data_set)),
                                 dtype=np.int)
        key_guesses = util.load_csv('{}/{}/Value/key_guesses_ALL_transposed.csv'.format(
            args.traces_path,
            str(load_args.data_set)),
            delimiter=' ',
            dtype=np.int,
            start=load_args.train_size + load_args.validation_size,
            size=load_args.attack_size)

    # Start a thread for each prediction
    groups_of = 7
    for k in range(math.ceil(len(predictions) / float(groups_of))):

        # Start groups of processes
        processes = []
        for i in range(k * groups_of, (k + 1) * groups_of, 1):
            if i >= len(predictions):
                break
            print(f"i: {i}")

            p = Process(target=threaded_run_test, args=(args, predictions[i], folder, args.run,
                                                        network_name, model_params, real_key, i))
            processes.append(p)
            p.start()

        # Wait for the processes to finish
        for p in processes:
            p.join()
            print('Joined process')


def threaded_run_test(args, prediction, folder, run, network_name, model_params, real_key, index_filter):
    # Shuffle the data using same permutation  for n_exp and calculate mean for GE of the model
    y = []
    for exp_i in range(args.num_exps):
        # Select permutation
        permutation = args.permutations[exp_i]

        # Shuffle data
        predictions_shuffled = shuffle_permutation(permutation, np.array(prediction))
        key_guesses_shuffled = shuffle_permutation(permutation, key_guesses)

        # Test the data
        x_exp, y_exp = test_with_key_guess_p(key_guesses_shuffled, predictions_shuffled,
                                             attack_size=args.attack_size,
                                             real_key=real_key,
                                             use_hw=args.use_hw)
        y.append(y_exp)

    # Calculate the mean over the experiments
    y = np.mean(y, axis=0)
    save_path = '{}model_r{}_{}_f{}.exp'.format(folder, run, get_save_name(network_name, model_params), index_filter)
    print("Save path {}".format(save_path))
    save_np(save_path, y, f="%f")


load_args = util.EmptySpace()
load_args.device = util.device
load_args.models_path = "/media/rico/Data/TU/thesis/runs3"
load_args.use_hw = False
load_args.traces_path = "/media/rico/Data/TU/thesis/data/"
load_args.raw_traces = False
load_args.train_size = 40000
load_args.validation_size = 1000
load_args.attack_size = 5000
load_args.use_noise_data = False
load_args.data_set = util.DataSet.RANDOM_DELAY_NORMALIZED
load_args.subkey_index = 2
load_args.desync = 0
load_args.unmask = True
load_args.spread_factor = 1
load_args.epochs = 75
load_args.batch_size = 100
load_args.lr = 0.0001
load_args.l2_penalty = 0.0005
load_args.init_weights = "kaiming"
load_args.num_exps = 20
load_args.permutations = util.generate_permutations(load_args.num_exps, load_args.attack_size)
load_args.run = 1

model_params = {
    "kernel_size": 100,
    "channel_size": 128,
    "max_pool": 64,
}

get_ranks(load_args, "SmallCNN", model_params, disable_filter4)
