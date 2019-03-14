import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from old.ascad import test_model, load_ascad

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

use_hw = False
sub_key_index = 2
n_classes = 9 if use_hw else 256
traces_file = '/media/rico/Data/TU/thesis/data/ASCAD.h5'
(x_profiling, y_profiling), (x_attack, y_attack), (metadata_profiling, metadata_attack) = load_ascad(traces_file,
                                                                                                     load_metadata=True)

if use_hw:
    y_profiling = [HW[val] for val in y_profiling]
    y_attack = [HW[val] for val in y_attack]
y_profiling = to_categorical(y_profiling, num_classes=n_classes, dtype='int32')
y_attack = to_categorical(y_attack, num_classes=n_classes, dtype='int32')

n_input = np.shape(x_profiling[1])[0]

gl_initializer = tf.glorot_uniform_initializer()
z_init = tf.zeros_initializer()

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# def mlp(x, num_classes):
#     layer_1 = tf.layers.dense(x, 200,
#                               activation=tf.nn.relu,
#                               bias_initializer=tf.zeros_initializer(),
#                               kernel_initializer=tf.glorot_uniform_initializer())
#     layer_2 = tf.layers.dense(layer_1, 200,
#                               activation=tf.nn.relu,
#                               bias_initializer=tf.zeros_initializer(),
#                               kernel_initializer=tf.glorot_uniform_initializer())
#     layer_3 = tf.layers.dense(layer_2, 200,
#                               activation=tf.nn.relu,
#                               bias_initializer=tf.zeros_initializer(),
#                               kernel_initializer=tf.glorot_uniform_initializer())
#     layer_4 = tf.layers.dense(layer_3, 200,
#                               activation=tf.nn.relu,
#                               bias_initializer=tf.zeros_initializer(),
#                               kernel_initializer=tf.glorot_uniform_initializer())
#     layer_5 = tf.layers.dense(layer_4, 200,
#                               activation=tf.nn.relu,
#                               bias_initializer=tf.zeros_initializer(),
#                               kernel_initializer=tf.glorot_uniform_initializer())
#     layer_6 = tf.layers.dense(layer_5, num_classes,
#                               activation=None,
#                               bias_initializer=tf.zeros_initializer(),
#                               kernel_initializer=tf.glorot_uniform_initializer())
#     return layer_6


n_hidden = 200
weights = {
    'in': tf.Variable(gl_initializer([700, n_hidden])),
    'w2': tf.Variable(gl_initializer([n_hidden, n_hidden])),
    'w3': tf.Variable(gl_initializer([n_hidden, n_hidden])),
    'w4': tf.Variable(gl_initializer([n_hidden, n_hidden])),
    'w5': tf.Variable(gl_initializer([n_hidden, n_hidden])),
    'out': tf.Variable(gl_initializer([n_hidden, n_classes])),

}
biases = {
    'in': tf.Variable(z_init([n_hidden])),
    'b2': tf.Variable(z_init([n_hidden])),
    'b3': tf.Variable(z_init([n_hidden])),
    'b4': tf.Variable(z_init([n_hidden])),
    'b5': tf.Variable(z_init([n_hidden])),
    'out': tf.Variable(z_init([n_classes])),
}


def mlp_wb(x):
    input_layer = tf.nn.relu(tf.add(tf.matmul(x, weights['in']), biases['in']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(input_layer, weights['w2']), biases['b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['w3']), biases['b3']))
    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['w4']), biases['b4']))
    layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['w5']), biases['b5']))
    out_layer = tf.add(tf.matmul(layer_5, weights['out']), biases['out'])
    return out_layer


# Parameters
learning_rate = 0.00001
training_epochs = 200
batch_size = 100
display_step = 1

# logits = mlp(X, n_classes)
logits = mlp_wb(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))

# optimizer = tf.train.RMSPropOptimizer(
#     learning_rate=learning_rate,
#     decay=0.,
#     momentum=0.9,
#     epsilon=1e-7)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-7)


train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batches = int(50000 / batch_size)
        # Loop over all batches
        for i in range(total_batches):
            batch_x, batch_y = (x_profiling[i * batch_size: (i + 1) * batch_size],
                                y_profiling[i * batch_size: (i + 1) * batch_size])
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batches
        # Display logs per epoch step
        if epoch % display_step == 0:
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}, acc train set {:.9f}".format(avg_cost,
                                                                                             accuracy.eval(
                                                                                                 {X: x_profiling,
                                                                                                  Y: y_profiling})))
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
        for numTrace in range(10000):
            plain = metadata_attack[numTrace]['plaintext'][sub_key_index]
            for subKey in range(256):
                n = predictions[numTrace][HW[subKey]] / C8[subKey]
                index = SBOX_INV[subKey] ^ plain
                guess[index] += n
                # guess[subKey] += predictions[numTrace][z] / C8[subKey]
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
