import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam

from interpretable_utils_list import *
from interpretable_utils_matrix import *


class Attr_Binary_FF_list(Model):
    """
    -- Attribute-interpretable Feed-forward neural network --
    uses a list implementation.

    Typical parameter object:
        params = {
            'input_dim': (2,),
            'hidden_layers_dims': [16, 8, 4],
            'activation': 'relu',

            'input_node_partition': [(0, 1), (1, 2)],
            'attribution_subsets':  [(True, False), (False, True), (True, True)],
        }

    Don't hesitate to copy / modify the dense_block function that is applied to
    each layer item of the list `z` to create more elaborate networks.
    """
    def __init__(self, params):
        super(Attr_Binary_FF_list, self).__init__()
        self.build_model(params)


    @staticmethod
    def dense_block(x, params):
        ''' Your usual dense layer.
        * params = {
                F: layer size,
                A: activation,
                name: group layer name,
                batch_norm: boolean,
                dropout_rate: None or float between 0 and 1
            }
        '''
        y = Dense(params['F'],
            activation=params['A'],
            name='{}_dense_{}'.format(params['name'], params['i']))(x)
        if 'batch_norm' in params and params['batch_norm']:
            y = BatchNormalization()(y)
        if 'dropout_rate' in params and params['dropout_rate']:
            y = Dropout(params['dropout_rate'])(y)
        return y


    @staticmethod
    def out_block(x, params):
        ''' Output layer, with the formulation of the paper. '''
        y = Dense(1, activation='tanh',
                name='output_{}'.format(params['i']))(x)
        return y


    def build_model(self, params):
        assert len(params['input_dim']) in [1, 2]
        is_temporal = len(params['input_dim']) == 3
        if type(params['activation']) == str:
            activation = tf.keras.activations.get(params['activation'])
        else:
            activation = params['activation']

        # Create the interpretable subset inputs
        x = Input(params['input_dim'], name='input')
        x_subsets = split_input(x,
                        params['attribution_subsets'],
                        params['input_node_partition'],
                        temporal=is_temporal)  # partition + gather

        # Defines the experts connection matrices
        inclusion_matrix = generate_tr(params['attribution_subsets'])
        C_tr_ = tf.convert_to_tensor(inclusion_matrix, 'float32')
        self.C_residual = C_tr_ - tf.eye(C_tr_.shape[0])

        # Hidden layers
        z = x_subsets
        for i, F in enumerate(params['hidden_layers_dims']):
            z = apply_layer(z, self.dense_block, { 'F': F,
                                                'A': activation,
                                                'name': 'hidden_' + str(i) })
            z = gather_int(z, inclusion_matrix)

        # Output
        v = apply_layer(z, self.out_block, {})
        v_concat = Concatenate(axis=-1)(v)
        v_out = Lambda(lambda x: MoE(x, self.C_residual, mode='L2'),
                       name='MoE')(v_concat)
        v_out = Lambda(lambda x: (x + 1.) / 2)(v_out)
        self.model_int = Model(inputs=x, outputs=v_concat)
        self.model = Model(inputs=x, outputs=v_out)


    def compile(self, opt):
        super(Attr_Binary_FF_list, self).compile()
        self.opt = opt
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.Accuracy(name='train_accuracy')


    def call(self, x):
        return self.model(x)


    def train_step(self, data):
        eps = 1e-6  # -- for log instabilities
        x, y = data
        if len(y.shape) == 1:
            y = tf.expand_dims(y, -1)  # caused a strange bug in the loss
        p = self.model(x)

        with tf.GradientTape() as tape:
            p_concat = self.model_int(x, training=True)

            # E-step
            mask = get_attributions(p_concat,
                                    self.C_residual,
                                    mode='L2',
                                    multiply_self = False)
            mask = mask / (tf.reduce_sum(mask, axis=-1, keepdims=True) + eps)

            # M-step
            red_p = (1. + p_concat) / 2.
            m_step_loss = - tf.reduce_mean(
                                tf.reduce_sum((
                                    y * tf.math.log(eps + red_p) +
                                    (1 - y) * tf.math.log(eps + 1 - red_p)
                                ) * mask, axis=-1)
                            )

        grads = tape.gradient(m_step_loss, self.model_int.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model_int.trainable_weights))

        self.train_loss.update_state(m_step_loss)
        self.train_acc.update_state(y, tf.round(p))
        return {'loss': self.train_loss.result(),
                'acc': self.train_acc.result() }

    @property
    def metrics(self):
        return [self.train_loss, self.train_acc]
