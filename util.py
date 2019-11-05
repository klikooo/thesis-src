import argparse
import itertools
from decimal import Decimal

import numpy as np
import sys
import h5py
import os.path
import torch
from enum import Enum

device = torch.device('cuda:0')
req_kernel_size = ['ConvNetKernel', 'ConvNetKernelAscad', 'ConvNetKernelMasked', 'ConvNetKernelAscad2']

SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])
SBOX_INV = np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])

HW = [bin(x).count("1") for x in range(256)]
C8 = [HW.count(HW[x]) / 256 for x in range(256)]


def HD(x, y):
    return bin(x ^ y).count("1")


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    x_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    # Load profiling labels
    y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    x_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    # Load attacking labels
    y_attack = np.array(in_file['Attack_traces/labels'])
    if not load_metadata:
        return (x_profiling, y_profiling), (x_attack, y_attack)
    else:
        return (x_profiling, y_profiling), (x_attack, y_attack), \
               (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])


def test_model(predictions, metadata, sub_key_index, use_hw=False, rank_step=10, unmask=False):
    real_key = metadata[0]['key'][sub_key_index]
    min_trace_idx = 0
    num_traces = len(metadata)

    ranks = full_ranks(predictions, real_key, metadata, min_trace_idx,
                       num_traces, rank_step, sub_key_index, use_hw, unmask)
    # We plot the results
    x = [ranks[i][0] for i in range(0, ranks.shape[0])]
    y = [ranks[i][1] for i in range(0, ranks.shape[0])]
    return x, y


def full_ranks(predictions, real_key, metadata, min_trace_idx, max_trace_idx, rank_step, sub_key_index, use_hw,
               unmask=False):
    index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    f = rank_hw if use_hw else rank
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = f(predictions[t - rank_step:t], metadata, real_key, t - rank_step, t,
                                           key_bytes_proba, sub_key_index, unmask)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks


def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, sub_key_index,
         unmask=False):
    # TODO: use unmask to unmask the data as with rank_hw

    # Compute the rank
    if len(last_key_bytes_proba) == 0:
        # If this is the first rank we compute, initialize all the estimates to zero
        key_bytes_proba = np.zeros(256)
    else:
        # This is not the first rank we compute: we optimize things by using the
        # previous computations to save time!
        key_bytes_proba = last_key_bytes_proba

    for p in range(0, max_trace_idx - min_trace_idx):
        # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
        plaintext = metadata[min_trace_idx + p]['plaintext'][sub_key_index]
        if unmask:
            mask = metadata[min_trace_idx + p]['masks'][sub_key_index - 2]
        else:
            mask = 0
        for i in range(0, 256):
            # Our candidate key byte probability is the sum of the predictions logs
            proba = predictions[p][SBOX[plaintext ^ i] ^ mask]
            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                # We do not want an -inf here, put a very small epsilon
                # that correspondis to a power of our min non zero proba
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba ** 2)
    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return real_key_rank, key_bytes_proba


def rank_hw(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, sub_key_index,
            unmask=False):
    # Compute the rank
    if len(last_key_bytes_proba) == 0:
        # If this is the first rank we compute, initialize all the estimates to zero
        key_bytes_proba = np.zeros(256)
    else:
        # This is not the first rank we compute: we optimize things by using the
        # previous computations to save time!
        key_bytes_proba = last_key_bytes_proba

    for p in range(0, max_trace_idx - min_trace_idx):
        # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
        plaintext = metadata[min_trace_idx + p]['plaintext'][sub_key_index]
        if unmask:
            mask = metadata[min_trace_idx + p]['masks'][sub_key_index - 2]
            # real_key = real_key ^ mask
        else:
            mask = 0
        for i in range(0, 256):
            # Our candidate key byte probability is the sum of the predictions logs

            # Original:
            # j = i ^ mask
            # proba = predictions[p][HW[j]] / C8[j]
            # index = SBOX_INV[j] ^ plaintext

            index = i
            proba = predictions[p][HW[SBOX[plaintext ^ i] ^ mask]]

            if proba != 0:
                key_bytes_proba[index] += np.log(proba)
            else:
                # We do not want an -inf here, put a very small epsilon
                # that correspondis to a power of our min non zero proba
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[index] += np.log(min_proba ** 2)
    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return real_key_rank, key_bytes_proba


def shuffle_permutation(permutation, to_shuffle):
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = to_shuffle[permutation]
    return shuffled_a


def save_model(network, model_save_file):
    # Make sure the path where the model is saved is stored
    os.makedirs(os.path.dirname(model_save_file), exist_ok=True)
    network.save(model_save_file)


class BoolAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs)
        self.default = kwargs['default']
        if nargs is not None:
            raise ValueError("nargs not allowed")

    def __call__(self, parser, namespace, values, option_string=None):
        if not (values in ['True', 'true', '1'] or values in ['False', 'false', '0']):
            print("aaaaaaaaaaaaaaaaaaaaaa")
            raise ValueError("arg should be either true or false")
        setattr(namespace, self.dest, values in ['True', 'true', '1'])


def load_csv(file, delimiter=',', dtype=np.float, start=None, size=None):
    if size is None:
        return np.genfromtxt(file, delimiter=delimiter, dtype=dtype)
    elif start is None and size is not None:
        with open(file) as t_in:
            return np.genfromtxt(itertools.islice(t_in, size), delimiter=delimiter, dtype=dtype)
    elif start is not None and size is not None:
        with open(file) as t_in:
            return np.genfromtxt(itertools.islice(t_in, start, start + size), delimiter=delimiter, dtype=dtype)
    else:
        raise ValueError('Error loading data set')


def load_ascad_train_traces(args):
    print(args)
    traces_file = '{}/ASCAD/ASCAD_{}_desync{}.h5'.format(args['traces_path'], args['sub_key_index'], args['desync'])
    print('Loading {}'.format(traces_file))
    (x_train, y_train), (_, _), (metadata_profiling, _) = load_ascad(traces_file, load_metadata=True)

    plain = None
    if args['domain_knowledge']:
        plain = metadata_profiling[:]['plaintext'][:, args['sub_key_index']]
        if args['use_hw']:
            plain = np.array([HW[val] for val in plain])
        plain = hot_encode(plain, 9 if args['use_hw'] else 256, dtype=np.float)

    if args['unmask']:
        y_train = np.array(
            [y_train[i] ^ metadata_profiling[i]['masks'][args['sub_key_index'] - 2] for i in range(len(y_train))])
        # [y_profiling[i] ^ metadata_profiling[i]['masks'][15] for i in range(len(y_profiling))])

    # Convert values to hamming weight if asked for
    if args['use_hw']:
        y_train = np.array([HW[val] for val in y_train])

    return x_train, y_train, plain


def load_ascad_test_traces(args):
    print(args)
    traces_file = '{}/ASCAD/ASCAD_{}_desync{}.h5'.format(args['traces_path'], args['sub_key_index'], args['desync'])
    print('Loading {}'.format(traces_file))
    (_, _), (x_test, y_test), (_, metadata_attack) = load_ascad(traces_file, load_metadata=True)

    plain = metadata_attack[:]['plaintext'][:, args['sub_key_index']]

    if args['unmask']:
        y_test = np.array(
            [y_test[i] ^ metadata_attack[i]['masks'][args['sub_key_index'] - 2] for i in range(len(y_test))])
        # [y_profiling[i] ^ metadata_profiling[i]['masks'][15] for i in range(len(y_profiling))])

    # Convert values to hamming weight if asked for
    if args['use_hw']:
        y_test = np.array([HW[val] for val in y_test])

    key = metadata_attack[0]['key'][args['sub_key_index']]
    key_guesses = np.load('{}/ASCAD/key_guesses_{}masked_0.npy'.format(
        args['traces_path'],
        'un' if args['unmask'] else ''
    ))
    return x_test, y_test, plain, key, key_guesses


def load_aes_hd(args):
    print(args)
    hw = 'HW' if args['use_hw'] else 'Value'
    x_train = load_csv('{}/AES_HD/traces/traces_50_{}.csv'.format(args['traces_path'], hw),
                       delimiter=' ',
                       start=args.get('start'),
                       size=args.get('size'))
    y_train = load_csv('{}/AES_HD/{}/model.csv'.format(args['traces_path'], hw),
                       delimiter=' ',
                       dtype=np.long,
                       start=args.get('start'),
                       size=args.get('size'))
    return x_train, y_train, []


def load_dpav4(args):
    print(args)
    hw = 'HW' if args['use_hw'] else 'Value'
    if args['raw_traces']:
        x_train = load_csv('{}/DPAv4/traces/traces_complete.csv'.format(args['traces_path']),
                           delimiter=' ',
                           start=args.get('start'),
                           size=args.get('size'))
    else:
        x_train = load_csv('{}/DPAv4/traces/traces_50_{}.csv'.format(args['traces_path'], hw),
                           delimiter=' ',
                           start=args.get('start'),
                           size=args.get('size'))

    y_train = load_csv('{}/DPAv4/{}/model.csv'.format(args['traces_path'], hw),
                       delimiter=' ',
                       dtype=np.long,
                       start=args.get('start'),
                       size=args.get('size'))
    return x_train, y_train, None


def load_dpa_npy(args):
    print(args)

    # x_train = np.load('{}/DPAv4/traces/traces_complete.csv.npy'.format(args['traces_path']))
    x_train = np.load('{}/DPAv4/traces/50000_traces.npy'.format(args['traces_path']))
    x_train = x_train[args['start']:args['start'] + args.get('size')]
    import gc
    gc.collect()

    print("Loaded and cut x")
    y_train = np.load('{}/DPAv4/Value/model.csv.npy'.format(args['traces_path']))
    y_train = y_train[args['start']:args['start'] + args.get('size')]
    if args['use_hw']:
        y_train = np.array([HW[int(y_train[i])] for i in range(len(y_train))])
    else:
        y_train = np.array([int(y_train[i]) for i in range(len(y_train))])
    import gc
    gc.collect()
    return x_train, y_train, None


def load_random_delay(args):
    print(args)
    hw = 'HW' if args['use_hw'] else 'Value'
    if args['raw_traces']:
        f = "traces_complete"
        if 'use_noise_data' in args and args['use_noise_data']:
            f = "test"
        x_train = load_csv('{}/Random_Delay/traces/{}.csv'.format(args['traces_path'], f),
                           delimiter=' ',
                           start=args.get('start'),
                           size=args.get('size'))
    else:
        x_train = load_csv('{}/Random_Delay/traces/traces_50_{}.csv'.format(args['traces_path'], hw),
                           delimiter=' ',
                           start=args.get('start'),
                           size=args.get('size'))

    y_train = load_csv('{}/Random_Delay/{}/model.csv'.format(args['traces_path'], hw),
                       delimiter=' ',
                       dtype=np.long,
                       start=args.get('start'),
                       size=args.get('size'))
    if args['domain_knowledge']:
        # plain = load_csv('{}/Random_Delay/{}/plain_0.csv'.format(args['traces_path'], hw),
        #                  delimiter=' ',
        #                  dtype=np.int,
        #                  start=args.get('start'),
        #                  size=args.get('size'))
        # plain = hot_encode(plain, 9 if args['use_hw'] else 256, dtype=np.float)
        # TODO: fix this for domain knowledge
        plain = None
    else:
        plain = None
    return x_train, y_train, plain


def load_random_delay_npy(args):
    print(args)
    x_train = np.load('{}/Random_Delay/traces/traces_complete.csv.npy'.format(args['traces_path']))
    y_train = np.load('{}/Random_Delay/Value/model.csv.npy'.format(args['traces_path']))

    x_train = x_train[args['start']:args['start'] + args.get('size')]
    y_train = y_train[args['start']:args['start'] + args.get('size')]
    if args['use_hw']:
        y_train = np.array([HW[int(y_train[i])] for i in range(len(y_train))])

    y_train = np.reshape(y_train, (args.get('size')))
    return x_train, y_train, None


def load_random_delay_large(args):
    print(args)
    traces_step = 20000
    total_steps = np.math.ceil((args['start'] + args['size']) / traces_step)
    x_train = np.zeros((args['size'], 6250))
    y_train = np.zeros((args['size']))
    start_step = int(args['start'] / traces_step)

    index_start = 0
    for step in range(start_step, total_steps):
        x_file = '{}/Random_Delay_Large/traces/traces_{}.csv.npy'.format(args['traces_path'], traces_step * (step + 1))
        y_file = '{}/Random_Delay_Large/Value/model_{}.csv.npy'.format(args['traces_path'], traces_step * (step + 1))
        x = np.load(x_file)
        y = np.load(y_file)

        # Begin step
        if step == start_step:
            # There is only one step
            if step == total_steps - 1:
                x_train[0:args['size']] = x[args['start'] % traces_step:(args['start'] + args['size']) % traces_step]
                y_train[0:args['size']] = y[args['start'] % traces_step:(args['start'] + args['size']) % traces_step]
            # More steps to come
            else:
                x_train[0:traces_step - (args['start'] % traces_step)] = x[(args['start'] % traces_step):traces_step]
                y_train[0:traces_step - (args['start'] % traces_step)] = y[(args['start'] % traces_step):traces_step]
                index_start = traces_step - (args['start'] % traces_step)
        # Last step
        elif step == total_steps - 1:
            x_train[index_start:args['size']] = x[0:args['size'] - index_start]
            y_train[index_start:args['size']] = y[0:args['size'] - index_start]
        # More steps to come
        else:
            x_train[index_start:index_start + traces_step] = x[0:traces_step]
            y_train[index_start:index_start + traces_step] = y[0:traces_step]
            index_start += traces_step
    return x_train, y_train, None


def load_random_delay_large_key_guesses(traces_path, start, size):
    traces_step = 20000
    total_steps = np.math.ceil((start + size) / traces_step)
    key_guesses = np.zeros((size, 256))
    start_step = int(start / traces_step)

    index_start = 0
    for step in range(start_step, total_steps):
        file = '{}/Random_Delay_Large/Value/key_guesses_{}.csv.npy'.format(traces_path, traces_step * (step + 1))
        step_key_guesses = np.load(file)

        # Begin step
        if step == start_step:
            if step == total_steps - 1:
                key_guesses[0:size] = step_key_guesses[start % traces_step:(start + size) % traces_step]
            # More steps to come
            else:
                key_guesses[0:traces_step - (start % traces_step)] = step_key_guesses[(start % traces_step):traces_step]
                index_start = traces_step - (start % traces_step)
        # Last step
        elif step == total_steps - 1:
            key_guesses[index_start:size] = step_key_guesses[0:size - index_start]
        # More steps to come
        else:
            key_guesses[index_start:index_start + traces_step] = step_key_guesses[0:traces_step]
            index_start += traces_step
    return key_guesses.astype(np.int)


def load_data_generic(args):
    print(args)

    x_train_file = '{}/{}/traces/traces_complete.csv.npy'.format(args['traces_path'], str(args['data_set']))
    if args['use_noise_data']:
        x_train_file = '{}/{}/traces/traces_noise_{}.npy'.format(
            args['traces_path'], str(args['data_set']), args['noise_level'])
    print(f"Loading {x_train_file}")
    x_train = np.load(x_train_file)

    y_train = np.load('{}/{}/Value/model.csv.npy'.format(args['traces_path'], str(args['data_set'])))

    x_train = x_train[args['start']:args['start'] + args.get('size')]
    y_train = y_train[args['start']:args['start'] + args.get('size')]

    y_train = np.reshape(y_train, (args.get('size')))
    return x_train, y_train, None


def load_ascad_keys(args):
    print(args)

    path = f"{args['traces_path']}/{str(args['data_set'])}/"
    x_train_file = f'{path}traces/train_traces.npy'
    print(f"Loading {x_train_file}")
    x_train = np.load(x_train_file)

    y_file = '{}/Value/train_model{}_{}masked.csv.npy'.format(
        path,
        '_hw' if args['use_hw'] else '',
        'un' if args['unmask'] else ''
    )
    print(f"Loading y file {y_file}")
    y_train = np.load(y_file)

    plaintexts = np.load(f"{path}/Value/train_plaintexts.npy")

    x_train = x_train[args['start']:args['start'] + args.get('size')]
    y_train = y_train[args['start']:args['start'] + args.get('size')]
    plaintexts = plaintexts[args['start']:args['start'] + args.get('size')]
    if args['use_hw']:
        plaintexts = [HW[plaintexts[i]] for i in range(len(plaintexts))]
    plaintexts = hot_encode(plaintexts, 9 if args['use_hw'] else 256, dtype=np.float)

    y_train = np.reshape(y_train, (args.get('size')))

    return x_train, y_train, plaintexts


def load_ascad_keys_test(args):
    print(args)
    path = f"{args['traces_path']}/{str(args['data_set'])}/"

    x_test = None
    y_test = None
    plaintexts = None
    if args['load_traces']:
        x_test_file = f'{path}/traces/test_traces.npy'
        print(f"Loading {x_test_file}")
        x_test = np.load(x_test_file)

        y_file = '{}/Value/test_model{}_{}masked.csv.npy'.format(
            path,
            '_hw' if args['use_hw'] else '',
            'un' if args['unmask'] else ''
        )
        print(f"Loading y file {y_file}")
        y_test = np.load(y_file)

        x_test = x_test[0:args.get('size')]
        y_test = y_test[0:args.get('size')]
        y_test = np.reshape(y_test, (args.get('size')))
        if args['use_hw']:
            y_test = np.array([HW[y_test[i]] for i in range(len(y_test))])

        plaintexts = np.load(f"{path}/Value/test_plaintexts.npy")
        plaintexts = plaintexts[0:args.get('size')]
        # plaintexts = hot_encode(plaintexts, 9 if args['use_hw'] else 256, dtype=np.float)

    key_guesses_file = '{}/Value/key_guesses_{}masked.csv.npy'.format(
        path,
        'un' if args['unmask'] else ''
    )
    key_guesses = np.load(key_guesses_file)

    return x_test, y_test, plaintexts, 34, key_guesses


def load_ascad_normalized(args):
    print(args)
    x_train = np.load('{}/{}/traces/traces_normalized_t{}_v{}_{}{}.csv.npy'.format
                      (args['traces_path'], str(args['data_set']),
                       args['train_size'], args['validation_size'], args['desync'],
                       '' if not args['use_noise_data'] else
                       '' if args['noise_level'] <= 0 else f"_noise{args['noise_level']}"))
    y_train = np.load('{}/{}/Value/model_{}masked.npy'.format(args['traces_path'], str(args['data_set']),
                                                              'un' if args['unmask'] else ''))

    x_train = x_train[args['start']:args['start'] + args.get('size')]
    y_train = y_train[args['start']:args['start'] + args.get('size')]

    y_train = np.reshape(y_train, (args.get('size')))
    if args['use_hw']:
        y_train = np.array([HW[val] for val in y_train])
    return x_train, y_train, None


def load_ascad_normalized_test_traces(args):
    print(args)
    path = f"{args['traces_path']}/{str(args['data_set'])}/"

    x_test = None
    y_test = None
    if args['load_traces']:
        if args['use_noise_data'] and args['noise_level'] > 0:
            x = np.load('{}/traces/traces_normalized_t{}_v{}_{}{}.csv.npy'.format(
                path,
                args['train_size'], args['validation_size'], args['desync'],
                f"_noise{args['noise_level']}"))
            x_test = x
        else:
            x = np.load('{}/traces/traces_normalized_t{}_v{}_{}.csv.npy'.format(
                path,
                args['train_size'], args['validation_size'], args['desync']))
            x_test = x[args['start']:args['start'] + args['size']]

        y = np.load('{}/Value/model_{}masked.npy'.format(path,
                                                         'un' if args['unmask'] else ''))
        y_test = y[50000:50000 + args['size']]
        print("y shape {}".format(y.shape))

        # Convert values to hamming weight if asked for
        if args['use_hw']:
            y_test = np.array([HW[val] for val in y_test])

    key_guesses = np.load('{}/Value/key_guesses_{}masked.npy'.format(path,
                                                                     'un' if args['unmask'] else ''))

    return x_test, y_test, None, 224, key_guesses


def load_train_data_set_keys(args):
    print(args)
    path = f'{args["traces_path"]}/{str(args["data_set"])}/'
    x_train = np.load('{}/{}/traces/train_traces.npy'.format
                      (args['traces_path'], str(args['data_set'])))
    y_train = np.load('{}/{}/Value/train_model.npy'.format(args['traces_path'], str(args['data_set'])))
    plain = np.load(f'{path}/Value/train_plain.npy')

    x_train = x_train[args['start']:args['start'] + args.get('size')]
    y_train = y_train[args['start']:args['start'] + args.get('size')]
    plain = plain[args['start']:args['start'] + args.get('size')]

    if args['use_hw']:
        y_train = np.array([HW[int(val)] for val in y_train])
    return x_train, y_train, plain


def load_train_portability(args):
    print(args)
    path = f'{args["traces_path"]}/{str(args["data_set"])}/'
    x_train = np.load('{}/traces/traces_selected_Exp_3.npy'.format(path))
    y_train = np.load('{}/Value/model_Exp_3.npy'.format(path))

    x_train = x_train[args['start']:args['start'] + args.get('size')]
    y_train = y_train[args['start']:args['start'] + args.get('size')]

    if args['use_hw']:
        y_train = np.array([HW[int(val)] for val in y_train])
    return x_train, y_train, None


def load_test_portability(args):
    print(args)
    path = f'{args["traces_path"]}/{str(args["data_set"])}/'

    x_train = None
    y_train = None
    if args['load_traces']:
        x_train = np.load('{}/traces/traces_selected_Exp_7.npy'.format(path))
        y_train = np.load('{}/Value/model_Exp_3.npy'.format(path))
        x_train = x_train[0:args.get('size')]
        y_train = y_train[0:args.get('size')]
        if args['use_hw']:
            y_train = np.array([HW[int(val)] for val in y_train])

    # key_guesses = np.load(f'{path}/Value/test_key_guesses.npy')
    # key_guesses = key_guesses[0:args.get('size')]
    # key = np.load(f'{path}/Value/test_keys.npy')
    key_guesses = []
    key = 0

    return x_train, y_train, None, key, key_guesses


def load_test_data_set_keys(args):
    print(args)
    path = f'{args["traces_path"]}/{str(args["data_set"])}/'

    x_train = None
    y_train = None
    plain = None
    if args['load_traces']:
        x_train = np.load('{}/traces/test_traces.npy'.format(path))
        y_train = np.load('{}/Value/test_model.npy'.format(path))
        x_train = x_train[0:args.get('size')]
        y_train = y_train[0:args.get('size')]
        if args['use_hw']:
            y_train = np.array([HW[int(val)] for val in y_train])

        plain = np.load(f'{path}/Value/test_plain.npy')
        plain = plain[0:args.get('size')]

    key_guesses = np.load(f'{path}/Value/test_key_guesses.npy')
    key_guesses = key_guesses[0:args.get('size')]
    key = np.load(f'{path}/Value/test_keys.npy')

    return x_train, y_train, plain, key[0], key_guesses


def load_sim_mask_test_traces(args):
    print(args)

    x = np.load('{}/{}/traces/traces_complete.csv.npy'.format
                (args['traces_path'], str(args['data_set'])))
    y = np.load('{}/{}/Value/model.csv.npy'.format(args['traces_path'], str(args['data_set'])))
    key_guesses = np.load('{}/{}/Value/key_guesses_ALL_transposed.csv.npy'.format(args['traces_path'],
                                                                                  str(args['data_set'])))

    x_test = x[args['start']:args['start'] + args['size']]
    y_test = y[args['start']:args['start'] + args['size']]
    # print("y shape {}".format(y.shape))

    # Convert values to hamming weight if asked for
    # if args['use_hw']:
    #     y_test = np.array([HW[val] for val in y_test])
    return x_test, y_test, None, 23, key_guesses


def load_random_delay_dk(args):
    print(args)

    x_train = load_csv('{}/Random_Delay_DK/traces/traces.csv'.format(args['traces_path']),
                       delimiter=' ',
                       start=args.get('start'),
                       size=args.get('size'))

    y_train = load_csv('{}/Random_Delay_DK/Value/model.csv'.format(args['traces_path']),
                       delimiter=' ',
                       dtype=np.long,
                       start=args.get('start'),
                       size=args.get('size'))
    plain = load_csv('{}/Random_Delay_DK/Value/plaintexts.csv'.format(args['traces_path']),
                     delimiter=' ',
                     dtype=np.long,
                     start=args.get('start'),
                     size=args.get('size'))
    plain = hot_encode(plain, 9 if args['use_hw'] else 256, dtype=np.float)
    return x_train, y_train, plain


def load_test_rdn(args):
    print(args)
    path = f'{args["traces_path"]}/{str(args["data_set"])}/'

    x_test = None
    y_test = None
    plain = None
    if args['load_traces']:
        x_test, y_test, plain = load_data_generic(args)
    print('Loading key guesses')

    key_guesses = load_csv('{}/Value/key_guesses_ALL_transposed.csv'.format(path),
                           delimiter=' ',
                           dtype=np.int,
                           start=args['train_size'] + args['validation_size'],
                           size=args['attack_size'])
    key = load_csv('{}/secret_key.csv'.format(path),
                   dtype=np.int)

    return x_test, y_test, plain, key, key_guesses


class DataSet(Enum):
    ASCAD = 1
    AES_HD = 2
    DPA_V4 = 3
    RANDOM_DELAY = 4
    RANDOM_DELAY_LARGE = 5
    RANDOM_DELAY_DK = 6
    RANDOM_DELAY_NORMALIZED = 7
    ASCAD_NORMALIZED = 8
    SIM_MASK = 9
    ASCAD_KEYS = 10
    ASCAD_KEYS_NORMALIZED = 11
    ASCAD_NORM = 12
    KEYS = 13
    KEYS_1B = 14
    KEYS_1 = 15
    PORTABILITY = 16

    def __str__(self):
        if self.value == 1:
            return "ASCAD"
        elif self.value == 2:
            return "AES_HD"
        elif self.value == 3:
            return "DPAv4"
        elif self.value == 4:
            return "Random_Delay"
        elif self.value == 5:
            return "Random_Delay_Large"
        elif self.value == 6:
            return "Random_Delay_DK"
        elif self.value == 7:
            return "Random_Delay_Normalized"
        elif self.value == 8:
            return "ASCAD_Normalized"
        elif self.value == 9:
            return "Simulated_Mask"
        elif self.value == 10:
            return "ASCAD_Keys"
        elif self.value == 11:
            return "ASCAD_Keys_Normalized"
        elif self.value == 12:
            return "ASCAD_NORM"
        elif self.value == 13:
            return "KEYS"
        elif self.value == 14:
            return "KEYS_1B"
        elif self.value == 15:
            return "KEYS_1"
        elif self.value == 16:
            return "portability"
        else:
            print("ERROR {}".format(self.value))

    @staticmethod
    def from_string(s):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()


def hot_encode(vector, num_classes, dtype=np.int):
    return np.eye(num_classes)[vector].astype(dtype)


def func_in_list(func, l):
    for f in l:
        if f == func:
            return True
    return False


def save_np(path, data, f="%i"):
    np.savetxt(path, data, delimiter=' ', fmt=f)


def generate_permutations(n, size):
    permutations = []
    for i in range(n):
        permutations.append(np.random.permutation(size))
    return permutations


def loop_at_least_once(data, func, do):
    i = 0
    while True:
        if len(data) != 0:
            func(data[i])
        do()
        i += 1
        if i >= len(data):
            break


def loop_at_least_once_with_arg(data, func, do, arg):
    i = 0
    while True:
        if len(data) != 0:
            func(data[i])
        do(arg)
        i += 1
        if i >= len(data):
            break


def save_loss_acc(path, filename, res):
    path = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)

    np.save("{}/{}.ta".format(path, filename), res[0])
    np.save("{}/{}.va".format(path, filename), res[1])
    np.save("{}/{}.tl".format(path, filename), res[2])
    np.save("{}/{}.vl".format(path, filename), res[3])


def load_loss_acc(file):
    ta = np.load("{}.ta.npy".format(file))
    va = np.load("{}.va.npy".format(file))
    tl = np.load("{}.tl.npy".format(file))
    vl = np.load("{}.vl.npy".format(file))
    return ta, va, tl, vl


def generate_folder_name(args):
    return '{}/subkey_{}/{}{}{}_SF{}_E{}_BZ{}_LR{}{}{}/train{}'.format(
        str(args.data_set),
        args.subkey_index,
        '' if args.unmask else 'masked/',
        '' if args.desync is 0 else 'desync{}/'.format(args.desync),
        'HW' if args.use_hw else 'ID',
        args.spread_factor,
        args.epochs,
        args.batch_size,
        '%.2E' % Decimal(args.lr),
        '' if np.math.ceil(args.l2_penalty) <= 0 else '_L2_{}'.format(args.l2_penalty),
        '' if not args.init_weights else '_{}'.format(args.init_weights),
        args.train_size,
    )


def get_memory():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def format_bytes(size):
    power = 2 ** 10
    n = 0
    power_labels = {0: '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n] + 'bytes'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EmptySpace(object):
    pass


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_raw_feature_size(the_data_set):
    switcher = {DataSet.RANDOM_DELAY: 3500,
                DataSet.DPA_V4: 3000,
                DataSet.RANDOM_DELAY_LARGE: 6250,
                DataSet.RANDOM_DELAY_DK: 3500,
                DataSet.RANDOM_DELAY_NORMALIZED: 3500,
                DataSet.ASCAD_NORMALIZED: 700,
                DataSet.SIM_MASK: 700,
                DataSet.ASCAD_KEYS: 1400,
                DataSet.ASCAD_KEYS_NORMALIZED: 1400,
                DataSet.ASCAD_NORM: 700,
                DataSet.ASCAD: 700,
                DataSet.KEYS: 500,
                DataSet.KEYS_1B: 500,
                DataSet.KEYS_1: 500,
                DataSet.PORTABILITY: 600}
    return switcher[the_data_set]


def load_data_set(data_set):
    table = {DataSet.ASCAD: load_ascad_train_traces,
             DataSet.AES_HD: load_aes_hd,
             DataSet.DPA_V4: load_dpa_npy,
             DataSet.RANDOM_DELAY: load_random_delay_npy,
             DataSet.RANDOM_DELAY_LARGE: load_random_delay_large,
             DataSet.RANDOM_DELAY_DK: load_random_delay_dk,
             DataSet.RANDOM_DELAY_NORMALIZED: load_data_generic,
             DataSet.ASCAD_NORMALIZED: load_ascad_normalized,
             DataSet.SIM_MASK: load_data_generic,
             DataSet.ASCAD_KEYS: load_ascad_keys,
             DataSet.ASCAD_KEYS_NORMALIZED: load_ascad_keys,
             DataSet.ASCAD_NORM: load_ascad_normalized,
             DataSet.KEYS: load_train_data_set_keys,
             DataSet.KEYS_1B: load_train_data_set_keys,
             DataSet.KEYS_1: load_train_data_set_keys,
             DataSet.PORTABILITY: load_train_portability,
             }
    return table[data_set]


def loader_test_data(data_set):
    switcher = {
        DataSet.ASCAD: load_ascad_test_traces,
        DataSet.KEYS: load_test_data_set_keys,
        DataSet.KEYS_1: load_test_data_set_keys,
        DataSet.KEYS_1B: load_test_data_set_keys,
        DataSet.ASCAD_NORMALIZED: load_ascad_normalized_test_traces,
        DataSet.ASCAD_NORM: load_ascad_normalized_test_traces,
        DataSet.SIM_MASK: load_sim_mask_test_traces,
        DataSet.ASCAD_KEYS: load_ascad_keys_test,
        DataSet.ASCAD_KEYS_NORMALIZED: load_ascad_keys_test,
        DataSet.RANDOM_DELAY_LARGE: load_test_random_delay_large,
        DataSet.RANDOM_DELAY_NORMALIZED: load_test_rdn,
        DataSet.AES_HD: load_test_generic,
        DataSet.DPA_V4: load_test_generic,
        DataSet.RANDOM_DELAY: load_test_generic,
        DataSet.PORTABILITY: load_test_portability,
    }
    return switcher[data_set]


def load_test_random_delay_large(args):
    loader = load_data_set(args.data_set)
    total_x_attack, total_y_attack, plain = loader({'use_hw': args.use_hw,
                                                    'traces_path': args.traces_path,
                                                    'raw_traces': args.raw_traces,
                                                    'start': args.train_size + args.validation_size,
                                                    'size': args.attack_size,
                                                    'domain_knowledge': True,
                                                    'use_noise_data': args.use_noise_data,
                                                    'data_set': args.data_set})
    print('Loading key guesses')
    data_set_name = str(args.data_set)
    _key_guesses = load_random_delay_large_key_guesses(args.traces_path,
                                                       args.train_size + args.validation_size,
                                                       args.attack_size)
    _real_key = load_csv('{}/{}/secret_key.csv'.format(args.traces_path, data_set_name),
                         dtype=np.int)

    _x_attack = total_x_attack
    _y_attack = total_y_attack
    return _x_attack, _y_attack, None, _real_key, _key_guesses


def load_test_generic(args):
    loader = load_data_set(args['data_set'])
    total_x_attack, total_y_attack, plain = loader(args)
    _dk_plain = None
    if plain is not None:
        _dk_plain = torch.from_numpy(plain).cuda()

    print(total_y_attack)
    print(np.shape(total_y_attack))
    print('Loading key guesses')

    ####################################
    # Load the key guesses and the key #
    ####################################
    data_set_name = str(args['data_set'])
    _key_guesses = load_csv('{}/{}/Value/key_guesses_ALL_transposed.csv'.format(
        args['traces_path'],
        data_set_name),
        delimiter=' ',
        dtype=np.int,
        start=args['train_size'] + args['validation_size'],
        size=args['attack_size'])
    _real_key = load_csv('{}/{}/secret_key.csv'.format(args['traces_path'], data_set_name),
                         dtype=np.int)

    _x_attack = total_x_attack
    _y_attack = total_y_attack
    return _x_attack, _y_attack, _dk_plain, _real_key, _key_guesses


def load_test_data(args):
    _x_attack, _y_attack, _real_key, _dk_plain, _key_guesses = None, None, None, None, None
    args = {'use_hw': args.use_hw,
            'traces_path': args.traces_path,
            'raw_traces': args.raw_traces,
            'start': args.train_size + args.validation_size,
            'size': args.attack_size,
            'train_size': args.train_size,
            'validation_size': args.validation_size,
            'domain_knowledge': True,
            'use_noise_data': args.use_noise_data,
            'data_set': args.data_set,
            'sub_key_index': args.subkey_index,
            'desync': args.desync,
            'unmask': args.unmask,
            'noise_level': args.noise_level,
            'load_traces': args.load_traces,
            'attack_size': args.attack_size}
    loader_function = loader_test_data(args['data_set'])
    return loader_function(args)


def w_print(msg):
    print(f"{BColors.WARNING}{msg}{BColors.ENDC}")


def e_print(msg):
    print(f"{BColors.FAIL}{msg}{BColors.ENDC}")


def cm(n):
    import matplotlib.cm as color_map
    return color_map.jet(np.linspace(1, 0, n))


def line_marker():
    return iter(('s', '+', '<', 'o', "D", "H", "*", ".", "^"))
