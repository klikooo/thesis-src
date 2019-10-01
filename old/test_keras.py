from keras import Model, Input
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling1D
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
import util


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


if __name__ == "__main__":
    hw = False
    x, y, _ = util.load_ascad_train_traces({
        "traces_path": "/media/rico/Data/TU/thesis/data/",
        "sub_key_index": 2,
        "desync": 0,
        "domain_knowledge": False,
        "use_hw": hw,
        "unmask": False,
    })

    x = x[0:40000]
    y = y[0:40000]

    num_classes = 9 if hw else 256
    y_profiling = to_categorical(y, num_classes=256, dtype='int32')
    save_model = ModelCheckpoint("test_model_cnn")
    callbacks = [save_model]

    x = x.reshape((x.shape[0], x.shape[1], 1))

    model = cnn_model(num_classes)
    model.fit(x, y_profiling, epochs=75, batch_size=256, callbacks=callbacks,
              validation_split=0.05)

