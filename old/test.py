import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 10  # 1st layer number of neurons
n_hidden_2 = 10  # 2nd layer number of neurons
n_hidden_3 = 10
n_input = 10  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
spread_factor = {
    's1': tf.Variable(tf.random_normal([n_hidden_2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
zeros = tf.zeros([n_hidden_2])
ones = tf.ones([n_hidden_2])
twos = tf.add(ones, ones)

x = list(np.random.rand(100))  # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
y = list(np.random.rand(100))  # [1.0, 3.0, 4.0, 7.0, 8.0, 9.0]
z = list(np.random.rand(100))  # [0.0, 0.0, 2.0, -1.0, 0.0, 13.0]
x = [x, y, z]

tf_min = tf.Variable([np.finfo(np.float32).max] * len(y))
tf_max = tf.Variable([np.finfo(np.float32).min] * len(y))

input_shape = len(y)
print('Input shape: {}'.format(input_shape))

_spread_factor = tf.constant([3])

config = tf.ConfigProto(device_count={'GPU': 0})
init = tf.global_variables_initializer()
with tf.Session(config=config) as session:
    session.run(init)

    tf_min = tf.minimum(tf_min, tf.reduce_min(x, axis=[0]))
    tf_max = tf.maximum(tf_max, tf.reduce_max(x, axis=[0]))

    tf_diff = tf_max - tf_min
    tf_teller = tf.subtract(x, tf_min)
    tf_noemer = tf.subtract(tf_max, tf_min)
    tf_prime = tf.divide(tf_teller, tf_noemer)
    tf_prime2 = tf.multiply(tf_prime, tf.to_float(_spread_factor))

    print('MIN: {}'.format(session.run(tf_min)))
    print('MAX {}'.format(session.run(tf_max)))
    print('LEN {}'.format(session.run(tf_diff)))
    print('NOEMER {}'.format(session.run(tf_noemer)))
    print('TELLER {}'.format(session.run(tf_teller)))
    print('X PRIME {}'.format(session.run(tf_prime)))
    print('X PRIME2 {}'.format(session.run(tf_prime2)))

    print('Data :{} '.format(x))
    x_grave = tf_prime2
    print('x_grave :{}'.format(session.run(x_grave)))

    print("shape div :{}".format(session.run(tf.shape(x_grave))))
    print("rank div :{}".format(session.run(tf.rank(x_grave))))

    # Duplicate x' more for spreading
    # Example:
    # [[a,b],[c,d]] -> [[a,b],[c,d],[a,b],[c,d]]
    x_grave = tf.reshape(x_grave, [-1])
    tile = tf.tile(x_grave, _spread_factor)
    tile = tf.reshape(tile, [_spread_factor[0], tf.shape(x_grave)[0]])

    # Reshape as follows:
    # [[a,b],[c,d],[a,b],[c,d]] -> [a,a,b,b,c,c,d,d]
    print('tile test {}'.format(session.run(tile)))
    tr = tf.transpose(tile, perm=[1, 0])
    print('tile transpose {}:'.format(session.run(tr)))
    x_grave_spreaded = tf.reshape(tr, [-1])
    print('tile reshape{}:'.format(session.run(x_grave_spreaded)))

    # Create centroids:
    # [0.5, 1.5, 0.5, 1.5 etc]
    half = tf.constant(0.5)
    step = tf.constant(1.0)
    end = tf.subtract(tf.to_float(_spread_factor)[0], half)
    bound = tf.reshape(_spread_factor, [])
    centroids = tf.tile(tf.range(half, bound, step), [tf.shape(x_grave)[0]])
    print('centroids {}'.format((session.run(centroids))))

    # Calculate max(0, 1 - abs(c - x'))
    num_features = tf.shape(centroids)[0]
    absolute = tf.abs(tf.subtract(centroids, x_grave_spreaded))
    right = tf.subtract(tf.ones(num_features), absolute)
    nc = tf.maximum(tf.zeros(num_features), right)
    print('nc: {}'.format(session.run(nc)))

    # Perform the function nc(x')
    result = tf.where(
        tf.logical_or(
            tf.logical_and(
                tf.equal(centroids, half),
                tf.less(x_grave_spreaded, centroids)),
            tf.logical_and(
                tf.equal(centroids, end),
                tf.greater(x_grave_spreaded, centroids))),
        tf.ones(tf.shape(x_grave_spreaded)[0]),
        nc)
    print(session.run(result))

    #
    num_spreaded_features = tf.multiply(_spread_factor[0], tf.shape(x)[1])
    spread_layer = tf.reshape(result, [tf.shape(x)[0], num_spreaded_features])
    print('out: {}'.format(session.run(spread_layer)))
    print('input shape: {}'.format(session.run(tf.shape(x))))
    print('out shape: {}'.format(session.run(tf.shape(spread_layer))))
    print('input shape: {}'.format(session.run(tf.shape(x)[0])))
    print('x_grave_spreaded shape: {}'.format(session.run(num_spreaded_features)))

    # spread_layer = tf.reshape(result, [num_spreaded_features, tf.shape(x)[0]])
    # print('out: {}'.format(session.run(tf.transpose(spread_layer))))

    #

    # print('centroid: {}'.format(session.run(centroid)))
    # print('layer_2: {}'.format(session.run(layer_2)))
    # print(session.run(tf.minimum(centroid, layer_2)))
