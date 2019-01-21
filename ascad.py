import h5py
import os.path
import sys
import numpy as np
from keras import Model, Input

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling1D
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from util import SBOX, HW, SBOX_INV, C8


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
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if not load_metadata:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), \
               (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])


def cnn_model(num_classes):
    input_shape = (700, 1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best')
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def mlp_model(num_classes):
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(700,)))  # 1
    model.add(Dense(200, activation='relu'))  # 2
    model.add(Dense(200, activation='relu'))  # 3
    model.add(Dense(200, activation='relu'))  # 4
    model.add(Dense(200, activation='relu'))  # 5
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(lr=0.00001)  # RMSprop(lr=0.00001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model(model_name, db, num_classes=256, batch_size=100, epochs=75, new=False):
    if new is False:
        try:
            check_file_exists(model_name)
            return load_model(model_name)
        except:
            pass

    (x_profiling, y_profiling), (x_attack, y_attack) = load_ascad(db)
    y_profiling = to_categorical(y_profiling, num_classes=num_classes, dtype='int32')
    y_attack = to_categorical(y_attack, num_classes=num_classes, dtype='int32')

    save_model = ModelCheckpoint(model_name)
    callbacks = [save_model]

    # model = spread_model(num_classes)
    model = mlp_model(num_classes)

    # num_traces = 500
    # x_profiling = x_profiling[:num_traces, :]
    # y_profiling = y_profiling[:num_traces, :]
    if len(model.get_layer(index=0).input_shape) == 3:
        x_profiling = x_profiling.reshape((x_profiling.shape[0], x_profiling.shape[1], 1))
    model.fit(x_profiling, y_profiling, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    return model


# ranks = full_ranks(model, X_attack, Metadata_attack, 0, num_traces, 10, sub_key_index)
# def full_ranks(model, dataset, metadata, min_trace_idx, max_trace_idx, rank_step, sub_key_index):


def test_model(predictions, metadata, sub_key_index, use_hw=False, title='Tensorflow', show_plot=True, rank_step=10
               , unmask=False):
    if predictions is None:
        predictions = model.predict(x_test)
    real_key = metadata[0]['key'][sub_key_index]
    min_trace_idx = 0
    num_traces = len(metadata)

    ranks = full_ranks(predictions, real_key, metadata, min_trace_idx
                       , num_traces, rank_step, sub_key_index, use_hw, unmask)
    # We plot the results
    x = [ranks[i][0] for i in range(0, ranks.shape[0])]
    y = [ranks[i][1] for i in range(0, ranks.shape[0])]

    if show_plot:
        plt.title('Performance of {}'.format(title))
        plt.xlabel('number of traces')
        plt.ylabel('rank')
        plt.grid(True)
        plt.plot(x, y)
        plt.show()
        plt.figure()
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


def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, sub_key_index
         , unmask=False):
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
        for i in range(0, 256):
            # Our candidate key byte probability is the sum of the predictions logs
            proba = predictions[p][SBOX[plaintext ^ i]]
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
            j = i ^ mask
            proba = predictions[p][HW[j]] / C8[j]
            index = SBOX_INV[j] ^ plaintext

            # index = SBOX_INV[i] ^ plaintext
            # proba = predictions[p][HW[index]] / C8[i]
            # index = SBOX_INV[i] ^ plaintext
            # else:
            #     proba = predictions[p][HW[i]] / C8[i]
            #     index = SBOX_INV[i] ^ plaintext

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


if __name__ == "__main__":
    for sub_key_index in range(2, 3):
        model_n = "_model_subkey_{}".format(sub_key_index)
        # file = '/Data/TU/thesis/src/data/ASCAD_data/ASCAD_databases/subkeys/ASCAD_subkey_{}'.format(sub_key_index)
        file = '/media/rico/Data/TU/thesis/data/ASCAD.h5'
        use_hw = True
        n_classes = 8 if use_hw else 256
        model = get_model(model_n, file, epochs=200, batch_size=100, new=True)

        (_, _), (x_test, y_test), (metadata_profiling, metadata_attack) = \
            load_ascad(file, load_metadata=True)
        predi = model.predict(x_test)
        test_model(predi, metadata_attack, sub_key_index, title='Keras')

# loss_and_metrics = model.evaluate(x_attack, y_attack, batch_size=200)
# (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) =
