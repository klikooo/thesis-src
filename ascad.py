
from keras import Model, Input

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling1D
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint

from util import SBOX, HW, load_ascad, check_file_exists, test_model




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

        x,y = test_model(predi, metadata_attack, sub_key_index)
        #TODO: plot result

