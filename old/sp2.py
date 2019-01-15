import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from ascad import test_model, load_ascad
from sklearn.model_selection import train_test_split


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
HW = [bin(x).count("1") for x in range(256)]


use_hw = False
sub_key_index = 2
n_classes = 9 if use_hw else 256
traces_file = '/media/rico/Data/TU/thesis/data/ASCAD.h5'
(x_profiling, y_profiling), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(traces_file,
                                                                                                     load_metadata=True)

# p = np.random.permutation(len(x))
# print(np.array(x)[p], np.array(y)[p])

if use_hw:
    y_profiling = [HW[val] for val in y_profiling]
    y_attack = [HW[val] for val in y_attack]
y_profiling = to_categorical(y_profiling, num_classes=n_classes, dtype='int32')
y_attack = to_categorical(y_attack, num_classes=n_classes, dtype='int32')

n_input = np.shape(x_profiling[1])[0]
n_neurons = 200
n_hidden_1 = 200  # 1st layer number of neurons
n_hidden_2 = 200  # 2nd layer number of neurons
n_hidden_3 = 200
n_hidden_4 = 200
n_hidden_5 = 200


# SPREAD VARS
tf_min = tf.Variable([np.finfo(np.float32).max] * 100, trainable=False)
tf_max = tf.Variable([np.finfo(np.float32).min] * 100, trainable=False)
_sf_ = 6
_spread_factor = tf.constant([_sf_])

gl_initializer = tf.glorot_uniform_initializer()
weights = {
    'w1': tf.Variable(gl_initializer([n_input, n_neurons])),
    'w2': tf.Variable(gl_initializer([100 * _sf_, n_hidden_1])),
    'w3': tf.Variable(gl_initializer([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(gl_initializer([n_hidden_2, n_classes])),

    'h1': tf.Variable(gl_initializer([n_input, n_hidden_1])),
    'h2': tf.Variable(gl_initializer([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(gl_initializer([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(gl_initializer([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(gl_initializer([n_hidden_4, n_hidden_5])),
    'mout': tf.Variable(gl_initializer([n_hidden_5, n_classes]))

}
z_init = tf.zeros_initializer()
biases = {
    'b1': tf.Variable(z_init([n_neurons])),
    'b2': tf.Variable(z_init([n_hidden_1])),
    'b3': tf.Variable(z_init([n_hidden_2])),

    'out': tf.Variable(z_init([n_classes])),

    'bi1': tf.Variable(z_init([n_hidden_1])),
    'bi2': tf.Variable(z_init([n_hidden_2])),
    'bi3': tf.Variable(z_init([n_hidden_3])),
    'bi4': tf.Variable(z_init([n_hidden_4])),
    'bi5': tf.Variable(z_init([n_hidden_5])),
    'mout': tf.Variable(z_init([n_classes]))

}

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])


def spread_network(x):
    # Use these as global so we can calculate the minumim and maximum of the entire data set
    global tf_min, tf_max
    # Dense layer
    layer_1 = tf.layers.dense(x, 100,
                              activation=tf.nn.relu,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_initializer=tf.glorot_uniform_initializer())  # input layer

    # Calculate the minimum and maximum for each neuron
    tf_min = tf.minimum(tf_min, tf.reduce_min(layer_1, axis=[0]))
    tf_max = tf.maximum(tf_max, tf.reduce_max(layer_1, axis=[0]))
    # Calculate x' which is in [0,1]
    tf_teller = tf.subtract(layer_1, tf_min)
    tf_noemer = tf.subtract(tf_max, tf_min)
    x_grave = tf.divide(tf_teller, tf_noemer)
    x_grave = tf.multiply(x_grave, tf.to_float(_spread_factor))


    # Duplicate x' for spreading
    x_grave = tf.reshape(x_grave, [-1])
    tile = tf.tile(x_grave, _spread_factor)
    tile = tf.reshape(tile, [_spread_factor[0], tf.shape(x_grave)[0]])
    tr = tf.transpose(tile, perm=[1, 0])
    x_grave_spreaded = tf.reshape(tr, [-1])


    # Create the centroids for each neuron
    half = tf.constant(0.5)
    step = tf.constant(1.0)
    bound = tf.reshape(_spread_factor, [])
    centroids = tf.tile(tf.range(half, bound, step), [tf.shape(x_grave)[0]])

    # Calculate the new value
    num_features = tf.shape(centroids)[0]
    absolute = tf.abs(tf.subtract(centroids, x_grave_spreaded))
    right = tf.subtract(tf.ones(num_features), absolute)
    nc = tf.maximum(tf.zeros(num_features), right)

    # Reshape the vector
    num_spreaded_features = tf.multiply(_spread_factor[0], tf.shape(layer_1)[1])
    num_traces = tf.shape(layer_1)[0]
    spread_layer = tf.reshape(nc, [num_traces, num_spreaded_features])
    # spread_layer = tf.Print(spread_layer, [tf.shape(spread_layer)], "Spread layer shape: ")

    # Next hidden layer
    layer_3 = tf.add(tf.matmul(spread_layer, weights['w2']), biases['b2'])
    layer_3 = tf.nn.relu(layer_3)
    #
    layer_4 = tf.add(tf.matmul(layer_3, weights['w3']), biases['b3'])
    layer_4 = tf.nn.relu(layer_4)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    # out_layer = tf.nn.relu(out_layer)
    return out_layer


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['bi1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['bi2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['bi3'])
    layer_3 = tf.nn.relu(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['bi4'])
    layer_4 = tf.nn.relu(layer_4)

    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['bi5'])
    layer_5 = tf.nn.relu(layer_5)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(tf.matmul(layer_5, weights['mout']), biases['mout'])
    return out_layer


def mlp(x):
    layer_1 = tf.layers.dense(x, n_hidden_1,
                              activation=tf.nn.relu,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_initializer=tf.glorot_uniform_initializer())  # input layer
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.relu,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_initializer=tf.glorot_uniform_initializer())
    layer_3 = tf.layers.dense(layer_2, n_hidden_3, activation=tf.nn.relu,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_initializer=tf.glorot_uniform_initializer())
    layer_4 = tf.layers.dense(layer_3, n_hidden_4, activation=tf.nn.relu,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_initializer=tf.glorot_uniform_initializer())
    layer_5 = tf.layers.dense(layer_4, n_hidden_5, activation=tf.nn.relu,
                              bias_initializer=tf.zeros_initializer(),
                              kernel_initializer=tf.glorot_uniform_initializer())
    # layer_6 = tf.layers.dense(layer_5, n_hidden_6, activation=tf.nn.relu)
    out = tf.layers.dense(layer_5, n_classes,
                          bias_initializer=tf.zeros_initializer(),
                          kernel_initializer=tf.glorot_uniform_initializer())  # out_layer
    return out


# Parameters
learning_rate = 0.00001
training_epochs = 200
batch_size = 500
display_step = 1

logits = spread_network(X)
# logits = multilayer_perceptron(X)
# logits = mlp(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))

optimizer = tf.train.RMSPropOptimizer(
    learning_rate=learning_rate,
    epsilon=1e-7)  # tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(50000 / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = (x_profiling[i * batch_size: (i + 1) * batch_size],
                                y_profiling[i * batch_size: (i + 1) * batch_size])
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: x_attack, Y: y_attack}))

    predictions = sess.run(pred, feed_dict={X: x_attack})
    test_model(predictions, metadata_attack, sub_key_index, use_hw=use_hw)

    if use_hw:
        guess = np.zeros(256)
        for numTrace in range(6000):
            plain = metadata_attack[numTrace]['plaintext'][sub_key_index]
            for subKey in range(256):
                z = HW[SBOX[plain ^ subKey]]
                guess[subKey] += predictions[numTrace][z]
            if numTrace % 200 == 0:
                print('Guess at trace {}, (of sub key {}) is {}, Real key?: {}'.format(numTrace,
                                                                                       sub_key_index, np.argmax(guess),
                                                                                       metadata_attack[0]['key'][
                                                                                           sub_key_index]))
    else:
        guess = np.zeros(256)
        for numTrace in range(6000):
            plain = metadata_attack[numTrace]['plaintext'][sub_key_index]
            for subKey in range(256):
                z = SBOX[plain ^ subKey]
                guess[subKey] += predictions[numTrace][z]
            if numTrace % 200 == 0:
                print('Guess at trace {}, (of sub key {}) is {}, Real key?: {}'.format(numTrace,
                                                                                       sub_key_index, np.argmax(guess),
                                                                                       metadata_attack[0]['key'][
                                                                                           sub_key_index]))
