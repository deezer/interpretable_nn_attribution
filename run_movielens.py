"""
Code for the movielens experiments.
"""

import argparse
import os
import time
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, Reshape, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from movielens.dataset import MovieLensDataset1M
from models.interpretable_utils_list import *
from models.interpretable_utils_matrix import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset link", type=str, default="data/ml-1m")
parser.add_argument("--debug", help="reduced dataset mode", action='store_true')
parser.add_argument("--train", help="train or test", dest="train", action='store_true', default=True)
parser.add_argument("--test", help="train or test", dest="train", action='store_false')

parser.add_argument("--save_weights", help="whether to save the weights during training", action='store_true')
parser.add_argument("--save_dir", help="save directory path", type=str, default='')
parser.add_argument("--bs", help="training batch_size", type=int, default=256)
parser.add_argument("--run_baseline", help="whether to run the non-interpretable counter-part", action='store_true', default=False)
parser.add_argument("--model_name", help="model name for saving after training or loading for testing", default='movielens_' + str(int(time.time())))

parser.add_argument("--F", help="model hidden layer and embedding size, corresponding to the NeuCF architecture", type=int, default=64)
parser.add_argument("--div_F", help="hidden layer divider, relative to the embedding size", type=int, default=1)
parser.add_argument("--mult_F", help="hidden layer multiplier, relative to the embedding size", type=int, default=1)
parser.add_argument("--cb_F", help="content-based embedding size multiplier", type=int, default=2)

parser.add_argument("--baseline_mult_F", help="baseline model hidden layer multiplier, relative to the embedding size", type=int, default=4)

args = parser.parse_args()

dataset = MovieLensDataset1M(args.dataset, debug=args.debug)
dataset.split_implicit_data()

BATCH_SIZE = args.bs
MODE = 7          # all features, cf get_aug_implicit_data documentation
RUN_BASELINE = args.run_baseline
MODEL_OUT_NAME = args.model_name
SAVE = args.save_weights
DIR = args.save_dir
TRAIN = args.train


## model definition (based on the dataset object)

class Aug_NCF_MLP:
    ''' NeuCF MLP but with additional data '''
    def __init__(self, mode, F = 64, DF = 1, G = 2):
        self.build(mode, F, DF, G)
        self.name = "aug_MLP_" + str(mode)

    def build(self, mode, F, DF, G):
        model_inputs = []
        first_layer = []

        if mode & 1 > 0:
            # -- Xu
            Xu = Input(1, dtype='int32', name='Xu')
            model_inputs.append(Xu)
            emb_u_layer = Embedding(dataset.Nu, F,  # dataset.Nu,  max(users) + 1
                        embeddings_initializer = 'random_normal',
                        input_length=1,
                        name="embed_user")
            emb_u = emb_u_layer(Xu)
            emb_u = Flatten()(emb_u)
            first_layer.append(emb_u)

        if mode & 2 > 0:
            # -- Xi
            Xi = Input(1, dtype='int32', name='Xi')
            model_inputs.append(Xi)
            emb_i_layer = Embedding(dataset.Ni, F,  # dataset.Ni, max(items) + 1
                        embeddings_initializer = 'random_normal',
                        input_length=1,
                        name="embed_item")
            emb_i = emb_i_layer(Xi)
            emb_i = Flatten()(emb_i)
            first_layer.append(emb_i)

        if mode & 4 > 0:
        # user embeddings  - 8, ~log_2(size dict)
            Xs, Xa, Xc = Input(1, dtype='int32', name='Xs'), Input(1, dtype='int32', name='Xa'), Input(1, dtype='int32', name='Xc')
            emb_sex = Embedding(2, 1 * G,  # dataset.Ni
                            embeddings_initializer = 'random_normal',
                            input_length=1,
                            name="embed_sex")
            emb_age = Embedding(dataset.Na, 3 * G,  # dataset.Ni
                            embeddings_initializer = 'random_normal',
                            input_length=1,
                            name="embed_age")
            emb_cat = Embedding(dataset.Nc, 4 * G,  # dataset.Ni
                            embeddings_initializer = 'random_normal',
                            input_length=1,
                            name="embed_cat")
            emb_s = emb_sex(Xs)
            emb_a = emb_age(Xa)
            emb_c = emb_cat(Xc)
            emb_s = Flatten()(emb_s)
            emb_a = Flatten()(emb_a)
            emb_c = Flatten()(emb_c)

            Xm = Input(dataset.Ng + 1, name='Xm', dtype='float32')
            emb_m = Dense(8 * G, activation='tanh')(Xm)

            model_inputs += [Xs, Xa, Xc, Xm]
            first_layer += [emb_s, emb_a, emb_c, emb_m]

        z = Concatenate(axis=-1)(first_layer)

        z = Dense(F * 2 * DF, activation='relu')(z)
        z = Dense(F * DF, activation='relu')(z)
        z = Dense(F * DF // 2, activation='relu')(z)
        z = Dense(F * DF // 4, activation='relu')(z)
        z = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform')(z)

        m = Model(inputs=model_inputs, outputs=z)  #Xu, Xi,
        m.compile(
            loss = 'binary_crossentropy',
            optimizer=Adam(1e-3),
            metrics=['acc'],
        )
        self.model = m
        self.model_out = m



class Int_NCF_MLP:
    ''' OURS: Interpretable augmented MLP model'''
    def __init__(self, params):
        self.name = "interp_MLP"
        self.build(params)

    @staticmethod
    def dense_block(x, params):
        y = Dense(params['F'],
                  activation='relu',
                  name='{}_dense_{}'.format(params['name'], params['i'])
                )(x)
        return y

    @staticmethod
    def out_block(x, params):
        y = Dense(1, activation='sigmoid')(x)
        return y

    def build(self, params):
        model_inputs = []
        first_layer = []
        S = params['S']
        TR = params['TR']
        CM = params['CM']
        F = params['F']
        dF = params['dF']
        DF = params['DF']
        G = params['cb_factor']

        self.name += '_' + str(F)

        # ------ INPUT
        # -- Xu <0>
        Xu = Input(1, dtype='int32', name='Xu')
        model_inputs.append(Xu)
        emb_u_layer = Embedding(params['Nu'], F,  # dataset.Nu,  max(users) + 1
                    embeddings_initializer = 'random_normal',
                    input_length=1,
                    name="embed_user")
        emb_u = Flatten()(emb_u_layer(Xu))
        first_layer.append(emb_u)

        # -- Xi <1>
        Xi = Input(1, dtype='int32', name='Xi')
        model_inputs.append(Xi)
        emb_i_layer = Embedding(params['Ni'], F,  # dataset.Ni, max(items) + 1
                    embeddings_initializer = 'random_normal',
                    input_length=1,
                    name="embed_item")
        emb_i = Flatten()(emb_i_layer(Xi))
        first_layer.append(emb_i)

        # -- Content-based <2>
        Xs, Xa, Xc = Input(1, dtype='int32', name='Xs'), Input(1, dtype='int32', name='Xa'), Input(1, dtype='int32', name='Xc')
        emb_sex = Embedding(2, 1 * G,  # dataset.Ni
                        embeddings_initializer = 'random_normal',
                        input_length=1,
                        name="embed_sex")
        emb_age = Embedding(params['Na'], 3 * G,  # dataset.Ni
                        embeddings_initializer = 'random_normal',
                        input_length=1,
                        name="embed_age")
        emb_cat = Embedding(params['Nc'], 4 * G,  # dataset.Ni
                        embeddings_initializer = 'random_normal',
                        input_length=1,
                        name="embed_cat")
        emb_s = Flatten()(emb_sex(Xs))
        emb_a = Flatten()(emb_age(Xa))
        emb_c = Flatten()(emb_cat(Xc))

        Xm = Input(params['Ng'] + 1, name='Xm', dtype='float32')
        emb_m = Dense(8 * G, activation='tanh')(Xm)

        agg_content_based = Concatenate(axis=-1, name='content_based')([emb_s, emb_a, emb_c, emb_m]) # size 16
        model_inputs += [Xs, Xa, Xc, Xm]
        first_layer.append(agg_content_based)

        # -- MODEL
        z0 = gather_input(first_layer, S)
        z1 = apply_layer(z0, self.dense_block, { 'F': (F * 2) // dF * DF, 'name': 'fc1' })
        z1_group = gather_int(z1, TR)
        z2 = apply_layer(z1_group, self.dense_block, { 'F': F // dF * DF, 'name': 'fc2' })
        z2_group = gather_int(z2, TR)
        z3 = apply_layer(z2_group, self.dense_block, { 'F': (F // 2) // dF * DF, 'name': 'fc3' })
        z3_group = gather_int(z3, TR)
        z4 = apply_layer(z3_group, self.dense_block, { 'F': (F // 4) // dF * DF, 'name': 'fc4' })
        z4_group = gather_int(z4, TR)

        v_int = apply_layer(z4_group, self.out_block, {})  # sigmoid
        v_concat = Concatenate(axis=-1)(v_int)
        v_out = Lambda(lambda x: MoE_movielens(x, CM))(v_concat)

        self.model_out = Model(inputs=model_inputs, outputs=v_out)

        self.model = Model(inputs=model_inputs, outputs=v_concat)
        self.model.compile(
            loss = self.model_loss(CM),
            optimizer=Adam(1e-3),
            metrics=['acc'],
        )


    def model_loss(self, mask_mat):
        @tf.function
        def loss(y, p_concat):
            eps = 1e-6
            mask = get_attributions_movielens(p_concat, mask_mat, multiply_self=False)
            n_mask = tf.reduce_sum(mask, axis=(0,1)) + 1e-9

            masked_bce = (-tf.reduce_sum(
                    (y * tf.math.log(eps + p_concat) +
                    (1 - y) * tf.math.log(eps + 1 - p_concat)) * mask,
                axis=0) / n_mask)   # mean along batch

            return tf.reduce_sum(masked_bce)
        return loss


## Instantiate baseline and interpretable counter-part

if RUN_BASELINE:
    m_baseline = Aug_NCF_MLP(MODE, F = args.F, DF = args.baseline_mult_F)

S = [[0, 0, 1],  # content-based
    [1, 0, 1],   # content + user embeddings
    [0, 1, 1],   # content + item embeddings
    [1, 1, 1],   # all features
    ]

S = [[bool(x) for x in l] for l in S]
TR = generate_tr(S)  # Here, TR_ij = S_i \subset S_j !
child_mat = tf.convert_to_tensor(TR, 'float32') - tf.eye(len(TR))
params = {
    'Nu': dataset.Nu,
    'Ni': dataset.Ni,
    'Na': dataset.Na,
    'Nc': dataset.Nc,
    'Ng': dataset.Ng,
    'TR': TR,           # connection matrix (transition)
    'CM': child_mat,    # strictly included connection matrix
    'S': S,             # \mathcal{S}
    'F': args.F,            # embedding size
    'dF': args.div_F,            # hidden layer divider
    'DF': args.mult_F,            # hidden layer multiplier
    'cb_factor': args.cb_F,     # aug data multiplier
}
m = Int_NCF_MLP(params)


## Eval utils

def scores(P, K = 10):
    pos_items = float(P[-1])
    neg_items = sorted(P[:-1].reshape(-1).tolist())
    rank = 1
    while (neg_items[-rank] > pos_items) and rank <= K + 1:
        rank += 1
    if rank <= K:
        return 1, 1. / np.log2(rank+1), rank
    return 0, 0, rank

def evaluate_model(model, dataset, eval_set='test_set', limit=-1):
    res = []
    item_set = set(dataset.movies.keys())
    u = 0
    for user in dataset.ratings:
        u += 1
        if limit > 0 and u > limit: break
        pos_i = random.sample(dataset.ratings[user][eval_set], 1)

        neg_item_set = item_set - (dataset.ratings[user]['train_set'].union(dataset.ratings[user]['val_set']).union(dataset.ratings[user]['test_set']))

        neg_is = random.sample(neg_item_set, 99)
        neg_is += pos_i

        X = []
        if MODE & 1 > 0:
            X.append(np.tile(np.reshape(user, (1, 1)), (len(neg_is), 1)))  # Xu
        if MODE & 2 > 0:
            X.append(np.array(neg_is).reshape((-1, 1)))  # Xi

        if MODE & 4 > 0:
            X.append(np.tile(np.reshape(dataset.ratings[user]['sex'], (1, 1)), (len(neg_is), 1)))
            X.append(np.tile(np.reshape(dataset.ratings[user]['age'], (1, 1)), (len(neg_is), 1)))
            X.append(np.tile(np.reshape(dataset.ratings[user]['cat'], (1, 1)), (len(neg_is), 1)))
            Xm = []
            for it in neg_is:
                Xm.append([dataset.movies[it]['year']]
                            + dataset.movies[it]['genres'])
            X.append(np.array(Xm, dtype='float32').reshape((-1, dataset.Ng + 1)))

        P = model.predict(X)
        res.append(scores(P))
    return np.mean(res, axis=0)


## Train

best_hr, best_ndcg, best_iter = -1, -1, -1
best_hr_baseline, best_ndcg_baseline, best_iter_baseline = -1, -1, -1
test_scores = 0
test_scores_baseline = 0

if TRAIN:
    for e in range(20):
        print("Epoch", e)
        X, Y = dataset.get_aug_implicit_data(D = 4, mode = MODE)

        if RUN_BASELINE:
            m_baseline.model.fit(X, Y,
                        epochs=1,
                        batch_size=BATCH_SIZE,
                        verbose=2,
                        shuffle=True,
                        )
        m.model.fit(X, Y,
                    epochs=1,
                    batch_size=BATCH_SIZE,
                    verbose=2,
                    shuffle=True,
                    )
        if RUN_BASELINE:
            hr, ndcg, rank = evaluate_model(m_baseline.model, dataset, eval_set='val_set')
            print('Val hr baseline', hr)
            if hr > best_hr_baseline:
                best_hr_baseline, best_ndcg_baseline, best_iter_baseline = hr, ndcg, e
                test_scores_baseline = evaluate_model(m_baseline.model, dataset, eval_set='test_set')

        hr, ndcg, rank = evaluate_model(m.model_out, dataset, eval_set='val_set')
        print('Val hr', hr)
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, e
            test_scores = evaluate_model(m.model_out, dataset, eval_set='test_set')
            if SAVE:
                m.model.save_weights(os.path.join(DIR, MODEL_OUT_NAME), overwrite=True)
else:
    m.model.load_weights(os.path.join(DIR, MODEL_OUT_NAME))

## Test

if RUN_BASELINE:
    last_test_scores = evaluate_model(m_baseline.model, dataset, eval_set='test_set')
    print('Baseline test scores', test_scores_baseline, 'overfit:', last_test_scores)

last_test_scores = evaluate_model(m.model_out, dataset, eval_set='test_set')
print('Model test scores', test_scores, 'overfit:', last_test_scores)