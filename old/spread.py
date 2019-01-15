import tensorflow as tf
import numpy as np


class Borg:
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state


class Spread(Borg):

    def __init__(self, data, spread_factor=3):
        super().__init__()
        self.input = data

        self.spread_factor = tf.constant([spread_factor])
        # self.min = tf.Variable([np.finfo(np.float32).max] * tf.shape(self.input)[1])
        # self.max = tf.Variable([np.finfo(np.float32).max] * tf.shape(self.input)[1])

        tf.get_variable('min')
        self.min = tf.Variable([np.finfo(np.float32).max] * 100, name="min", trainable=False)
        self.max = tf.Variable([np.finfo(np.float32).max] * 100, name="max", trainable=False)
        self.min = tf.Print(self.min, [self.min], "RESET MIN")

    def apply(self):
        # Calculate the minimum and maximum for each neuron
        self.min = tf.minimum(self.min, tf.reduce_min(self.input, axis=[0]))
        self.max = tf.maximum(self.max, tf.reduce_max(self.input, axis=[0]))

        # Calculate x' which is in [0, spread factor]
        tf_numerator = tf.subtract(self.input, self.min)
        tf_denominator = tf.subtract(self.max, self.min)
        x_grave = tf.divide(tf_numerator, tf_denominator)
        x_grave = tf.multiply(x_grave, tf.to_float(self.spread_factor))

        # Duplicate x' for spreading
        x_grave = tf.reshape(x_grave, [-1])
        tile = tf.tile(x_grave, self.spread_factor)
        tile = tf.reshape(tile, [self.spread_factor[0], tf.shape(x_grave)[0]])
        tr = tf.transpose(tile, perm=[1, 0])
        x_grave_spreaded = tf.reshape(tr, [-1])

        # Create the centroids for each neuron
        half = tf.constant(0.5)
        step = tf.constant(1.0)
        bound = tf.reshape(self.spread_factor, [])
        centroids = tf.tile(tf.range(half, bound, step), [tf.shape(x_grave)[0]])

        # Calculate the new value
        num_features = tf.shape(centroids)[0]
        absolute = tf.abs(tf.subtract(centroids, x_grave_spreaded))
        right = tf.subtract(tf.ones(num_features), absolute)
        nc = tf.maximum(tf.zeros(num_features), right)

        # Reshape the vector
        num_spreaded_features = tf.multiply(self.spread_factor[0], tf.shape(self.input)[1])
        num_traces = tf.shape(self.input)[0]
        spread_layer = tf.reshape(nc, [num_traces, num_spreaded_features])

        return spread_layer


def spread_network(input, spread_factor=3):
    # Use these as global so we can calculate the minumim and maximum of the entire data set
    global tf_min, tf_max
    tf_spread_factor = tf.constant([spread_factor])

    # Calculate the minimum and maximum for each neuron
    tf_min = tf.minimum(tf_min, tf.reduce_min(input, axis=[0]))
    tf_max = tf.maximum(tf_max, tf.reduce_max(input, axis=[0]))

    # Calculate x' which is in [0, spread factor]
    tf_teller = tf.subtract(input, tf_min)
    tf_noemer = tf.subtract(tf_max, tf_min)
    x_grave = tf.divide(tf_teller, tf_noemer)
    x_grave = tf.multiply(x_grave, tf.to_float(tf_spread_factor))

    # Duplicate x' for spreading
    x_grave = tf.reshape(x_grave, [-1])
    tile = tf.tile(x_grave, tf_spread_factor)
    tile = tf.reshape(tile, [tf_spread_factor[0], tf.shape(x_grave)[0]])
    tr = tf.transpose(tile, perm=[1, 0])
    x_grave_spreaded = tf.reshape(tr, [-1])

    # Create the centroids for each neuron
    half = tf.constant(0.5)
    step = tf.constant(1.0)
    bound = tf.reshape(tf_spread_factor, [])
    centroids = tf.tile(tf.range(half, bound, step), [tf.shape(x_grave)[0]])

    # Calculate the new value
    num_features = tf.shape(centroids)[0]
    absolute = tf.abs(tf.subtract(centroids, x_grave_spreaded))
    right = tf.subtract(tf.ones(num_features), absolute)
    nc = tf.maximum(tf.zeros(num_features), right)

    # Reshape the vector
    num_spreaded_features = tf.multiply(tf_spread_factor[0], tf.shape(layer_1)[1])
    num_traces = tf.shape(input)[0]
    spread_layer = tf.reshape(nc, [num_traces, num_spreaded_features])

    return spread_layer


