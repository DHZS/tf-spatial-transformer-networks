####################
# DATE: 2018-10-8
####################
"""Classification Net
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from nets.spatial_transformer_layer import SpatialTransformerLayer


class ClassificationNet(keras.Model):
    def __init__(self, st_out_dim, num_class, **kwargs):
        super().__init__(self, **kwargs)
        # localization net
        loc_net = keras.Sequential()
        loc_net.add(Flatten())
        loc_net.add(Dense(64, activation=tf.nn.relu))
        loc_net.add(Dense(64, activation=tf.nn.relu))
        # classification net
        self.st = SpatialTransformerLayer(loc_net, st_out_dim)
        self.flatten = Flatten()
        self.fc1 = Dense(64, activation=tf.nn.relu)
        self.fc2 = Dense(64, activation=tf.nn.relu)
        self.fc3 = Dense(num_class)

    def call(self, inputs, training=None, mask=None):
        transformed, theta = self.st(inputs)
        net = self.flatten(transformed)
        net = self.fc1(net)
        net = self.fc2(net)
        logits = self.fc3(net)
        return logits, transformed, theta

    def train(self, optimizer, x, y):
        with tf.GradientTape() as tape:
            logits, transformed, theta = self.__call__(x)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.variables)
        optimizer.apply_gradients(zip(grads, self.variables))
        return loss, tf.nn.softmax(logits), transformed, theta

    def accuracy(self, prediction, y):
        eq = tf.to_float(tf.equal(tf.argmax(prediction, axis=-1), tf.argmax(y, axis=-1)))
        return tf.reduce_mean(eq)
