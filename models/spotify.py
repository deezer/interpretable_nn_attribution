'''
Interpretable deep neural networks using boosted jointed restricted models
'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Concatenate, Flatten, Activation, BatchNormalization, LeakyReLU, Add, GRU, Reshape, Bidirectional, Embedding
from tensorflow.keras.optimizers import Adam

from interpretable_utils_list import *
from multi_head_attention import *


class Classifier:
    '''
    General classifier class with predefined layer blocks
    '''
    def __init__(self):
        pass
    
    @staticmethod
    def dense_block(x, params):
        y_dense = Dense(params['F'],
            name='{}_dense_{}'.format(params['name'], params['i']))(x)
        y = LeakyReLU()(y_dense)
        y = BatchNormalization()(y)
        y = Dropout(0.1)(y)
        return y

    @staticmethod
    def bigru_skip_block(x, params):
        ''' If you feel GRUish '''
        y_bigru = Bidirectional(
            GRU(params['F']),
            name='{}_bigru_{}'.format(params['name'], params['i']))(x)
        y_skip = Dense(2 * params['F'], name='{}_dense_{}'.format(params['name'], params['i']))(x)
        y = Add()([y_bigru, y_skip])
        y = BatchNormalization()(y)
        return y
    
    @staticmethod
    def selfattention_block(x, params):
        ''' Transformer self attention block applied to x '''
        p = Dense(params['F'], name='{}_d1_{}'.format(params['name'], params['i']))(x)
        
        r = MultiHeadAttention(params['H'], params['F'] // params['H'])([x, x])
        r = Add()([r, p])
        r = LayerNormalization()(r)

        t = Dense(params['F'], name='{}_ff1_{}'.format(params['name'], params['i']))(r)
        t = LeakyReLU()(t)
        t = Dense(params['F'], name='{}_ff2_{}'.format(params['name'], params['i']))(t)
        t = Add()([t, r])
        t = LayerNormalization()(t)
        return t
    
    @staticmethod
    def crossattention_block(xy, params):
        ''' Transformer cross attention block applied to x and y'''
        x, y = xy
        
        p = Dense(params['F'], name='{}_d1_{}'.format(params['name'], params['i']))(x)
        
        r = MultiHeadAttention(params['H'], params['F'] // params['H'])([x, x])
        r = Add()([r, p])
        r = LayerNormalization()(r)
        
        v = MultiHeadAttention(params['H'], params['F'] // params['H'])([r, y])
        v = Add()([v, r])
        v = LayerNormalization()(v)

        t = Dense(params['F'], name='{}_ff1_{}'.format(params['name'], params['i']))(v)
        t = LeakyReLU()(t)
        t = Dense(params['F'], name='{}_ff2_{}'.format(params['name'], params['i']))(t)
        t = Add()([t, v])
        t = LayerNormalization()(t)
        return t

    @staticmethod
    def out_block(x, params):
        y = Dense(1, activation='tanh')(x)
        return y



class SpotifyClassifier(Classifier):
    '''
    Classifier with adapted metrics for the spotify dataset
    '''
    def __init__(self):
        super().__init__()
    
    @staticmethod
    @tf.function
    def mean_square_diff(y, a):
        mask = tf.dtypes.cast(y >= -1, tf.float32)
        sum_mask = 1e-6 + tf.reduce_sum(mask) * 4  # for [-1, 1], divide by 2^2
        return tf.reduce_sum(tf.square(a - y) * mask) / sum_mask
    
    @staticmethod
    @tf.function
    def maa(y, p):
        ''' Mean average accuracy, refer to the paper '''
        mask = tf.dtypes.cast(y >= -1, tf.float32)
        rounded_pred = tf.round((p + 1.) / 2.) * 2. - 1
        acc = (1 - tf.abs((rounded_pred - y)) / 2.) * mask
        cum_acc = tf.cumsum(acc, axis=1) / tf.expand_dims(1 + tf.range(0., tf.shape(y)[1]), -1)
        return tf.reduce_sum(acc * cum_acc, axis=(0,1)) / tf.reduce_sum(mask, axis=(0, 1))
    
    @staticmethod
    @tf.function
    def partial_accuracy(y, y_pred):
        ''' Accuracy taking into account variable session lengths '''
        mask = tf.dtypes.cast(y >= -1, tf.float32)
        rounded_pred = tf.round((y_pred + 1.) / 2.) * 2. - 1
        l2_diff = tf.square((y - rounded_pred) / 2.) * mask
        acc = 1 - (tf.reduce_sum(l2_diff, axis=(0,1)) / (1e-6 + tf.reduce_sum(mask, axis=(0,1))))
        return acc
    
    @staticmethod
    @tf.function
    def first_accuracy(y, y_pred):
        ''' Accuracy on the first track prediction only '''
        y = tf.round(y[:,:1])
        y_pred = y_pred[:,:1]
        rounded_pred = tf.round((y_pred + 1.) / 2.) * 2. - 1
        l2_diff = tf.square((y - rounded_pred) / 2.)
        acc = 1 - tf.reduce_mean(l2_diff)
        return acc



class Int_Transformer(SpotifyClassifier):
    ''' Interpretable transformer '''
    def __init__(self, params):
        self.name = 'Interpretable Transformer'
        self.save_name = 'attr_tfrm'
        self.mask_func = 'L2'
        self.build(params)
    

    def model_loss(self, mask_mat):
        @tf.function
        def loss(y, p_concat):
            data_mask = tf.dtypes.cast(y >= -1, tf.float32)  # var length
            
            mask = get_attributions(p_concat, mask_mat, mode = self.mask_func, multiply_self=False)
            mask = data_mask * mask
            n_mask = tf.reduce_sum(mask, axis=(0,1,2)) + 1e-9
            
            eps = 1e-6
            red_p = (p_concat + 1.) / 2.  # [-1, 1] -> [0,1]
            red_y = (y + 1.) / 2.  # [-1, 1] -> [0,1]
            masked_bce = (-tf.reduce_sum(
                    (red_y * tf.math.log(eps + red_p)
                    + (1 - red_y) * tf.math.log(eps + 1 - red_p)) * mask,
                axis=(0,1)) / n_mask)
            
            return tf.reduce_sum(masked_bce)
        return loss
        
        
    def build(self, params):
        L = params['sess_length']
        F = params['hidden_dim']
        TH = params['heads']

        TR_L = params['interpretation']['left']['inclusion_mat']  # [[Boolean]]
        TR_R = params['interpretation']['right']['inclusion_mat']
        CH = params['interpretation']['left']['child_mat']
        
        # -- input
        track_embedder = Embedding(
            input_dim=params['track_embedding'].shape[0],
            output_dim=params['track_embedding'].shape[1],
            weights=[params['track_embedding']],
            trainable=False,
        )
        flatten_embedding = Reshape((L // 2, params['track_embedding'].shape[1]))
        
        x_l = Input((L // 2, params['input_dim']), name='X_left')
        x_l_ids = Input((L // 2, 1), name='X_left_ids')
        x_l_embed = flatten_embedding(track_embedder(x_l_ids))
        x_l_concat = Concatenate(axis=-1)([x_l, x_l_embed])
        split_x_l = split_input(x_l_concat,
                params['interpretation']['left']['S'],
                params['interpretation']['left']['node_range'],
                temporal=True, name_prefix='left_')
        
        x_r = Input((L // 2, 2), name='X_right')  # sess pos
        x_r_ids = Input((L // 2, 1), name='X_right_ids')
        x_r_embed = flatten_embedding(track_embedder(x_r_ids))
        x_r_concat = Concatenate(axis=-1)([x_r, x_r_embed])
        split_x_r = split_input(x_r_concat,
                params['interpretation']['right']['S'],
                params['interpretation']['right']['node_range'],
                temporal=True, name_prefix='right_')
        
        # -- intermediate layers
        z1 = apply_layer(split_x_l, self.selfattention_block, { 'F': F, 'H': TH, 'name': 'left1' })
        z1_group = gather_int(z1, TR_L)
        z2 = apply_layer(z1_group, self.selfattention_block, { 'F': F, 'H': TH, 'name': 'left2' })
        z2_group = gather_int(z2, TR_L)
        z3 = apply_layer(z2_group, self.selfattention_block, { 'F': F,  'H': TH, 'name': 'left3' })
        memory_left = z3
        
        dup_x_r = duplicate_input(split_x_r, params['interpretation']['merge'])
        
        v1 = apply_layer(zip(dup_x_r, memory_left), self.crossattention_block, { 'F': F, 'H': TH, 'name': 'right1' })
        v1_group = gather_int(v1, TR_L)
        v2 = apply_layer(zip(v1_group, memory_left), self.crossattention_block, { 'F': F, 'H': TH, 'name': 'right2' })
        v2_group = gather_int(v2, TR_L)
        v3 = apply_layer(zip(v2_group, memory_left), self.crossattention_block, { 'F': F, 'H': TH, 'name': 'right3' })
        v3_group = gather_int(v3, TR_L)
        out_layer = v3_group
        
        
        # -- output
        v_int = apply_layer(out_layer, self.out_block, {})   # tanh
        v_concat = Concatenate(axis=-1)(v_int)
        v_out = Lambda(lambda x: MoE(x, CH, self.mask_func))(v_concat)
        
        self.model_out = Model(inputs=[x_l_ids, x_l, x_r_ids, x_r], outputs=v_out)
        self.model_out.compile(
            metrics=[self.partial_accuracy, self.first_accuracy, self.maa]
        )
    
        self.model = Model(inputs=[x_l_ids, x_l, x_r_ids, x_r], outputs=v_concat)
        self.model.compile(
            loss = self.model_loss(CH),
            optimizer=Adam(1e-3),
            metrics=[self.partial_accuracy, self.maa],
        )
