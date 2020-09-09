'''
Interpretable model utilities.

Those functions are used to define interpretable models using lists of Layers.
The other way to do it is by using mask_constraints, as done
in interpretable_utils_matrix.py.
'''

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Lambda


# -- layer wrappers

def slice_node(node, temporal):
    '''
    Slices according to node.
    * node: Tuple(int, int)
    '''
    if temporal:
        def f(x):
            return x[:,:,node[0]:node[1]]
    else:
        def f(x):
            return x[:,node[0]:node[1]]
    return f


def gather_input(input_list, S, name_prefix=''):
    '''
    Concatenate layers according to S.
    * input_list: Keras.Layers[]
    * S: Boolean[][]
    '''
    Y = []
    for h, sh_mask in enumerate(S):
        Sh = []
        for i, xi in enumerate(input_list):
            if sh_mask[i]:
                Sh.append(xi)
        if len(Sh) > 1:
            sh = Concatenate(axis=-1, name=name_prefix+'S'+str(h))(Sh)
        else:
            sh = Sh[0]
        Y.append(sh)
    return Y


def split_input(input_layer, S, node_range, temporal=False, name_prefix=''):
    '''
    Takes the input and partition according to given ranges
    and interpretation subsets to create an input interpretation subsets.
    * input_input: Keras.Layers
    * S: Boolean[][]
    * node_range: Tuple(int, int)[]
    '''
    X = []   # X -> (X_i) -- group-sparsity
    for i, node in enumerate(node_range):
        X.append(Lambda(slice_node(node, temporal), name=name_prefix+'X'+str(i))(input_layer))

    Y = gather_input(X, S, name_prefix)  # (X_i) -> (S_j) -- structured-sparsity
    return Y


def duplicate_input(input_layer, S_merge):
    '''
    When merging interpretable submodels with different configs.

    At the moment, only deals with right elements being included in the
    left ones, and the duplicate pattern has to be defined by hand.
    I purposefully write i from 0 to H in S_merge for clarity.

    Used shape : S is a correspondance table from left to right [[a, b]]
        typically, [i, corresponding(i)]
    '''
    y_list = []
    for merge_pattern in S_merge:
        y_list.append(input_layer[merge_pattern[1]])
    return y_list


def gather_int(x_list, tr_mat):
    '''
    Regroup intermediates layers into the given interpretation groups
    applies a stop_gradient to avoid updating child models
    * x_list: Keras.Layer[]
    * tr_mat: Boolean[][]   --  true iff S_j \subset S_i
    '''
    assert len(x_list) == len(tr_mat)
    H = len(x_list)
    y_list = []
    for i in range(H):
        sub_x_list = []
        for j in range(H):
            if tr_mat[j][i]:  # is Sj in Si
                z = x_list[j]
                if i != j:
                    z = Lambda(tf.stop_gradient)(z)
                sub_x_list.append(z)
        if len(sub_x_list) > 1:
            y = Concatenate(axis=-1)(sub_x_list)
        else:
            y = sub_x_list[0]
        y_list.append(y)
    return y_list


def merge_int(layers):
    '''
    Concatenate intermediate layer features of same nature.
    For instance, with a given S, and model_a(X, S) = Keras.Layer[]
    and model_b(X, S) = Keras.Layer[], we merge the representations of
    model_a and model_b of corresponding nature (S_1 with S_1, S_2 with S_2...).
    That's where the `duplicate_input` function comes handy if we have S and S'
    with S \subset S', then we can first adapt the representation from S, and
    then merge everything.
    * layers: Keras.Layer[][]
    '''
    y_list = []
    K = len(layers)
    H = len(layers[0])
    for xi in zip(*layers):
        yi = Concatenate(axis=-1)(list(xi))
        y_list.append(yi)
    return y_list


def apply_layer(x_list, func, params):
    '''
    Apply a unique function(input, params) to each item of x_list.
    Typically used to create a block of a network.
    * x_list: Keras.Layer[]
    * func: (Keras.Layer, dict) -> Keras.Layer
    * params: dict   -- shared params
      `i` will be used to name the layers.

    Example of func:
        def dense_block(x, params):
            y_dense = Dense(params['F'],
                name='{}_dense_{}'.format(params['name'], params['i']))(x)
            y = LeakyReLU()(y_dense)
            return y
    '''
    y_list = []
    params['i'] = 0
    for x in x_list:
        y = func(x, params)
        y_list.append(y)
        params['i'] += 1
    return y_list


def apply_layers(x_list, layers, params):
    '''
    Same as apply_layer but with a specific function for each item.
    '''
    y_list = []
    for i, x in enumerate(x_list):
        params[i]['i'] = i
        y = layers[i](x, params[i])
        y_list.append(y)
    return y_list



# -- Mixture of Experts functions

def mask_L1(p):
    conf = tf.stop_gradient(tf.math.abs(p))
    return conf

def mask_L2(p):
    conf = tf.stop_gradient(tf.math.square(p))
    return conf

def mask_Lp(p, n = 4):
    conf2 = tf.stop_gradient(tf.math.pow(tf.math.abs(p), n))
    conf1 = tf.stop_gradient(tf.math.pow(tf.math.abs(p), n-1))
    conf = 2 * conf2 / (1 + conf1)
    return conf

def get_attributions(p, mask_mat, mode = 'L1', multiply_self = True):
    if mode == 'L1':
        conf = mask_L1(p)
    elif mode == 'L2':
        conf = mask_L2(p)
    else:
        conf = mask_Lp(p)
    mask = tf.map_fn(lambda x: tf.reduce_prod(1. - x * conf, axis=-1), tf.transpose(mask_mat))
    if tf.shape(mask).shape == 2:
        mask = tf.transpose(mask, [1, 0])
    else:
        mask = tf.transpose(mask, [1, 2, 0])
    if multiply_self:
        mask = mask * conf         # -- faster training, depends on the model
    return mask

def MoE(p, mask_mat, mode = 'L1'):
    mask = get_attributions(p, mask_mat, mode)
    y = tf.reduce_sum(mask * p, axis=-1, keepdims=True) / (1e-9 + tf.reduce_sum(mask, axis=-1, keepdims=True))
    return y


# -- movielens MoE

def get_attributions_movielens(p, mask_mat, n = 4, multiply_self = True):
    # conf = tf.stop_gradient(tf.abs(p - 0.3)) / 0.7

    conf_up = tf.stop_gradient(tf.pow(tf.abs(p - 0.3), n))
    conf_down = tf.stop_gradient(tf.pow(tf.abs(p - 0.3), n-1))
    conf = conf_up / (1 + conf_down) * (tf.pow(tf.abs(0.7), n-1) + 1) / tf.pow(tf.abs(0.7), n)

    mask = tf.map_fn(lambda x: tf.reduce_prod(1. - x * conf, axis=-1), tf.transpose(mask_mat))
    mask = tf.transpose(mask, [1, 0])
    if multiply_self:
        mask = mask * conf         # -- faster training
    return mask

def MoE_movielens(p, mask_mat):
    ''' Bidimensional '''
    mask = get_attributions_movielens(p, mask_mat)
    y = tf.reduce_sum(mask * p, axis=-1, keepdims=True) / (1e-9 + tf.reduce_sum(mask, axis=-1, keepdims=True))
    return y