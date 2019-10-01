from keras import Model, Input
from keras.models import Sequential
from keras.layers.convolutional import AveragePooling1D, MaxPooling1D, Conv1D
from keras.layers import Dropout, Dense, BatchNormalization, Flatten
from keras.optimizers import Adam

import util
from keras.utils import to_categorical


train_size = 40000
validation_size = 1000
num_classes = 256
channels = 32
input_shape = (700, 1)
batch_size = 256
epochs = 75

size = train_size + validation_size
args = {
    "data_set": util.DataSet.ASCAD,
    "traces_path": "/media/rico/Data/TU/thesis/data",
    "train_size": train_size,
    "validation_size": validation_size,
    "use_hw": False,
    "desync": 0,
    "use_noise_data": False,
    "unmask": False,
    "size": size,
    "sub_key_index": 2,
    "domain_knowledge": False
}
x, y, _ = util.load_ascad_train_traces(args)
x_train = x[0:train_size]
y_train = y[0:train_size]
x_validation = x[train_size:size]
y_validation = y[train_size:size]

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_validation = x_validation.reshape((x_validation.shape[0], x_validation.shape[1], 1))

y_train = to_categorical(y_train, num_classes=256, dtype='int32')
y_validation = to_categorical(y_validation, num_classes=256, dtype='int32')


model = Sequential()


model.add(Conv1D(channels, kernel_size=7, activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(channels*2, kernel_size=7, activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(channels*4, kernel_size=7, activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling1D(2))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_validation, y_validation)
                    )
