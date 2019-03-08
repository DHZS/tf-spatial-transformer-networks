####################
# DATE: 2018-10-8
####################
"""Spatial Transformer Layer
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
tf.enable_eager_execution(conf)


class SpatialTransformerLayer(keras.Model):
    def __init__(self, loc_net, output_size, init_theta=None, **kwargs):
        super().__init__(**kwargs)
        self.output_size = [int(x) for x in output_size[0:2]]  # output feature map size in (H, W) order.
        self.batch_size = 0
        self.regular_grid = None
        self.__BIAS_INIT_VALUE = [1, 0, 0, 0, 1, 0]  # Identity matrix
        # self.__BIAS_INIT_VALUE = [1, 0, 0]  # crop and translation
        # Localization net
        self.loc_net = loc_net
        # affine transformation
        self.out = tf.layers.Dense(6,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.constant_initializer(init_theta if init_theta else self.__BIAS_INIT_VALUE))

    def call(self, inputs, training=None):
        net = self.loc_net(inputs)
        net = tf.layers.flatten(net)
        theta = self.out(net)

        # grid generator
        u_h, u_w = self.output_size
        batch_size = theta.get_shape().as_list()[0]

        # attention
        # zeros = tf.zeros([batch_size])
        # affine_matrix = tf.stack([theta[:, 0], zeros, theta[:, 1], zeros, theta[:, 0], theta[:, 2]], axis=-1)
        # theta = affine_matrix

        theta = tf.reshape(theta, [-1, 2, 3])
        regular_grid = self._create_regular_grid(batch_size)
        # transform
        trans_grid = tf.matmul(theta, regular_grid)  # [N, 2, H*W]
        trans_grid = tf.reshape(trans_grid, [-1, 2, u_h, u_w])

        # bilinear interpolation
        out = self._bilinear_interpolation(inputs, trans_grid)
        return out, theta

    def compute_output_shape(self, input_shape):
        return [input_shape[0], *self.output_size, input_shape[-1]]

    def _create_regular_grid(self, batch_size):
        # cache regular grid
        if batch_size == self.batch_size:
            return self.regular_grid
        self.batch_size = batch_size

        h, w = self.output_size
        # y axis
        y = tf.expand_dims(tf.linspace(-1., 1., h), axis=-1)
        y = tf.reshape(tf.tile(y, [1, w]), [-1])
        # x axis
        x = tf.tile(tf.linspace(-1., 1., w), [h])
        # constant 1
        one = tf.ones([h * w], dtype=tf.float32)
        grid = tf.stack([y, x, one])
        # expand to batch
        grid = tf.tile(tf.expand_dims(grid, axis=0), [batch_size, 1, 1])
        self.regular_grid = tf.to_float(grid)
        return grid

    def _get_pixel_value_at_point(self, inputs, indices):
        y, x = indices
        batch, h, w = y.get_shape().as_list()[0: 3]

        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1))
        b = tf.tile(batch_idx, (1, h, w))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)

    def _bilinear_interpolation(self, inputs, trans_grid):
        """
        :param inputs: feature map V described in paper
        :param trans_grid: coordinates in V, the tensor shape is [N, 2, H, W]
        :return:
        """
        batch, v_h, v_w, c = inputs.get_shape().as_list()

        # sampling grid V
        y_s = trans_grid[:, 0, :, :]  # shape [N, u_h, u_w]
        x_s = trans_grid[:, 1, :, :]

        # rescale x, y to [0, w-1/h-1]
        y_s = (y_s + 1.) * 0.5 * (v_h - 1)
        x_s = (x_s + 1.) * 0.5 * (v_w - 1)

        # four nearest points
        y0 = tf.to_int32(tf.floor(y_s))
        y1 = y0 + 1
        x0 = tf.to_int32(tf.floor(x_s))
        x1 = x0 + 1

        # clip
        y0 = tf.clip_by_value(y0, 0, v_h - 1)
        y1 = tf.clip_by_value(y1, 0, v_h - 1)
        x0 = tf.clip_by_value(x0, 0, v_w - 1)
        x1 = tf.clip_by_value(x1, 0, v_w - 1)

        # get pixel value at four points in V
        v0 = self._get_pixel_value_at_point(inputs, [y0, x0])
        v1 = self._get_pixel_value_at_point(inputs, [y0, x1])
        v2 = self._get_pixel_value_at_point(inputs, [y1, x0])
        v3 = self._get_pixel_value_at_point(inputs, [y1, x1])

        # cast to float32
        y0, y1, x0, x1 = [tf.to_float(x) for x in [y0, y1, x0, x1]]

        # calculate weight
        w0 = (y1 - y_s) * (x1 - x_s)
        w1 = (y1 - y_s) * (x_s - x0)
        w2 = (y_s - y0) * (x1 - x_s)
        w3 = (y_s - y0) * (x_s - x0)

        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(x, axis=-1) for x in [w0, w1, w2, w3]]

        # sample
        out = tf.add_n([w0 * v0, w1 * v1, w2 * v2, w3 * v3])
        return out
