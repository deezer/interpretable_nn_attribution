'''
Multi-head attention layer + other utils class for the transformer model.
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Reshape, Layer
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import activations


class LayerNormalization(Layer):
    '''
    BatchNormalisation() was diverging -> simpler is better.
    '''

    def __init__(self, h=1, **kwargs):
        self.h = h
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.shape = input_shape
        self.reshape = Reshape((input_shape[1], self.h, input_shape[2] // self.h))
        self.reshape_back = Reshape((input_shape[1], input_shape[2]))
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, epsilon=1e-6):
        x = self.reshape(x)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        norm_x = self.reshape_back(norm_x)
        return norm_x


class MaskWeights(Constraint):
    def __init__(self, mask=[], **kwargs):
        self.mask = np.copy(mask, order=2)
        self.w_mask = tf.convert_to_tensor(self.mask, dtype='float32')
        super(MaskWeights, self).__init__(**kwargs)

    def __call__(self, w):
        return w * self.w_mask

    def get_config(self):
        return {'mask': self.mask }


class MultiHeadAttention(Layer):
    def __init__(self, h, units, multiply_heads=1, **kwargs):
        '''
        Output size : h * nh * units
        '''
        self.h = h
        self.nh = multiply_heads
        self.units = units
        self.dim_out = self.h * self.nh * self.units
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shapes):
        x_shape, y_shape = input_shapes
        assert len(x_shape) == len(y_shape) == 3

        self.q_net = Dense(self.dim_out, input_shape=(x_shape,), use_bias=False, name="q_dense")
        self.k_net = Dense(self.dim_out, input_shape=(y_shape,), use_bias=False, name="k_dense")
        self.v_net = Dense(self.dim_out, input_shape=(y_shape,), use_bias=False, name="v_dense")
        self.out_net = Dense(self.dim_out, name="head_combiner")

        self.attention = Lambda(lambda x: activations.softmax(
            tf.matmul(x[0], x[1], transpose_b=True) / (self.units ** -0.5)
        ), name='scale_dot_attention')
        self.dot_product = Lambda(lambda x: tf.matmul(x[0], x[1]), name='dot_product')

        self.head_split = Reshape((x_shape[1], self.h * self.nh, self.units))
        self.head_swap = Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))
        self.head_merge = Reshape((x_shape[1], self.dim_out))

        super(MultiHeadAttention, self).build(input_shapes)

    def call(self, z):
        x, y = z

        q = self.q_net(x)
        k = self.k_net(y)
        v = self.v_net(y)

        q = self.head_swap(self.head_split(q))
        k = self.head_swap(self.head_split(k))
        v = self.head_swap(self.head_split(v))

        A = self.attention([q, k])
        self.attention_output = A

        z = self.dot_product([A, v])
        z = self.head_merge(self.head_swap(z))
        z = self.out_net(z)

        return z

    def compute_output_shape(self, input_shapes):
        input_shape, _ = input_shapes
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.dim_out
        return tuple(output_shape)

    def get_config(self):
        config = {
            'h': self.h,
            'units': self.units,
            'multiply_heads': self.nh,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskedMultiHeadAttention(Layer):
    '''
    Interpretable multihead attention module implemented using
    masked weights.

    * h: number of heads ie. different learned attention combinaisons
    * dout: number of output hidden features, multiple of h
    * multiply_heads: more heads by input class for more attention combinaisons
    * mask: custom mask built with utils_matrix.generate_connection_graph()
    '''

    def __init__(self, h, dout, multiply_heads=1, mask=[], **kwargs):
        assert dout % (h * multiply_heads) == 0
        self.h = h
        self.dout = dout
        self.nh = multiply_heads
        self.dk = self.dout // (self.h * self.nh)
        self.mask = mask
        super(MaskedMultiHeadAttention, self).__init__(**kwargs)

    def build(self, z):
        input_shape, _ = z

        self.q_net = Dense(self.dout, input_shape=input_shape, use_bias=False,
                           kernel_constraint=MaskWeights(self.mask),
                           name="q_dense")
        self.k_net = Dense(self.dout, input_shape=input_shape, use_bias=False,
                           kernel_constraint=MaskWeights(self.mask),
                           name="k_dense")
        self.v_net = Dense(self.dout, input_shape=input_shape, use_bias=False,
                           kernel_constraint=MaskWeights(self.mask),
                           name="v_dense")

        self.out_net = Dense(self.dout,
                             kernel_constraint=MaskWeights(self.mask),
                             name="head_combiner")

        self.attention = Lambda(lambda x: activations.softmax(
            tf.matmul(x[0], x[1], transpose_b=True) / (self.dk ** -0.5)
        ),
                                name='scale_dot_attention')
        self.dot_product = Lambda(lambda x: tf.matmul(x[0], x[1]),
                                  name='dot_product')

        self.head_split = Reshape((input_shape[1], self.h * self.nh, self.dk))
        self.head_swap = Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))
        self.head_merge = Reshape((input_shape[1], self.dout))

        super(MaskedMultiHeadAttention, self).build(input_shape)

    def call(self, z):
        '''
        x : (?, l, d) -> (?, l, dout)
        q, k, v : (?, l, dout) -> (?, h, l, dout/h)
        A : q.k' -> (?, h, l, l)
        z : A * v : (?, h, l, dout/h) -> (?, l, dout)
        '''
        x, y = z

        q = self.q_net(x)
        k = self.k_net(y)
        v = self.v_net(y)

        q = self.head_swap(self.head_split(q))
        k = self.head_swap(self.head_split(k))
        v = self.head_swap(self.head_split(v))

        A = self.attention([q, k])
        self.attention_output = A

        z = self.dot_product([A, v])
        z = self.head_merge(self.head_swap(z))
        z = self.out_net(z)

        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dout,)
