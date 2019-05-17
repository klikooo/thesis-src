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
    (x_train, y_train), (_, _), (metadata_profiling, metadata_attack) = load_ascad(traces_file, load_metadata=True)

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
    if args['domain_knowledge']:
        # TODO : add plain 0
        if True:
            plain = []
        else:
            plain = load_csv('{}/DPAv4/{}/plain_0.csv'.format(args['traces_path'], hw),
                             delimiter=' ',
                             dtype=np.int,
                             start=args.get('start'),
                             size=args.get('size'))
            plain = hot_encode(plain, 9 if args['use_hw'] else 256, dtype=np.float)
    else:
        plain = None
    return x_train, y_train, plain


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

    y_train = y_train[:args.get('size')]
    y_train = np.reshape(y_train, (args.get('size')))
    return x_train[:args.get('size')], y_train, None


def load_random_delay_large(args):
    print(args)

    x_train = load_csv('{}/Random_Delay_Large/traces/traces.csv'.format(args['traces_path']),
                       delimiter=' ',
                       start=args.get('start'),
                       size=args.get('size'))

    y_train = load_csv('{}/Random_Delay_Large/Value/model.csv'.format(args['traces_path']),
                       delimiter=' ',
                       dtype=np.long,
                       start=args.get('start'),
                       size=args.get('size'))
    return x_train, y_train, None


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


class DataSet(Enum):
    ASCAD = 1
    AES_HD = 2
    DPA_V4 = 3
    RANDOM_DELAY = 4
    RANDOM_DELAY_LARGE = 5
    RANDOM_DELAY_DK = 6

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
        else:
            print("ERROR {}".format(self.value))

    @staticmethod
    def from_string(s):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()


def load_data_set(data_set):
    table = {DataSet.ASCAD: load_ascad_train_traces,
             DataSet.AES_HD: load_aes_hd,
             DataSet.DPA_V4: load_dpav4,
             DataSet.RANDOM_DELAY: load_random_delay_npy,
             DataSet.RANDOM_DELAY_LARGE: load_random_delay_large,
             DataSet.RANDOM_DELAY_DK: load_random_delay_dk}
    return table[data_set]


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
        '' if args.unmask or args.data_set is not DataSet.ASCAD else 'masked/',
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
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'


class EmptySpace(object):
    pass


