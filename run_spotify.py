"""
Code for the sequential skip prediction task
using the preprocessed spotify parquet files.
"""

import gc
import time
import sys
import glob
import numpy as np
import pandas as pd
sys.path.append('models')

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard

from skip_prediction.spotify_mock_generator import *
from models.interpretable_utils_matrix import *
from models.spotify import *


DATASET_PATH = 'spotify_parquet/'
TRACK_F_PATH = 'data/track_features.parquet'

AGG_GENERATORS = True
DEBUG = True

SPOTIFY_WEIGHTS_PATH = 'spotify_weights'
SPOTIFY_LOG_PATH = 'spotify_logs'
SPOTIFY_RESULTS_PATH = 'spotify_results'


## LOAD DATA

train_col = ['ordinal_track_id_clean',

    'session_position',   # 1
    'session_length',
    'hist_user_behavior_is_shuffle',
    'hour_of_day',
    'premium',
    'date_gap',
    'no_pause_before_play',
    'short_pause_before_play',
    'long_pause_before_play',
    'context_type_charts',
    'context_type_user_collection',
    'context_type_editorial_playlist',
    'context_type_personalized_playlist',
    'context_type_catalog',
    'context_type_radio',   # 15

    'hist_user_behavior_reason_start_appload',  # 16
    'hist_user_behavior_reason_start_popup',
    'hist_user_behavior_reason_start_trackerror',
    'hist_user_behavior_reason_start_backbtn',
    'hist_user_behavior_reason_start_playbtn',
    'hist_user_behavior_reason_start_remote',
    'hist_user_behavior_reason_start_uriopen',
    'hist_user_behavior_reason_start_trackdone',
    'hist_user_behavior_reason_start_clickrow',
    'hist_user_behavior_reason_start_fwdbtn',
    'hist_user_behavior_reason_start_endplay',
    'hist_user_behavior_reason_end_appload',
    'hist_user_behavior_reason_end_popup',
    'hist_user_behavior_reason_end_trackerror',
    'hist_user_behavior_reason_end_backbtn',
    'hist_user_behavior_reason_end_playbtn',
    'hist_user_behavior_reason_end_remote',
    'hist_user_behavior_reason_end_uriopen',
    'hist_user_behavior_reason_end_trackdone',
    'hist_user_behavior_reason_end_clickrow',
    'hist_user_behavior_reason_end_fwdbtn',
    'hist_user_behavior_reason_end_endplay',
    'skip_1',
    'skip_2',
    'skip_3',
    'not_skipped',
    'context_switch',
    'hist_user_behavior_n_seekfwd',
    'hist_user_behavior_n_seekback',  # 44
    ]

track_col = [
    'duration',  # 45
    'release_year',
    'us_popularity_estimate',

    'acousticness',  # 48
    'beat_strength',
    'bounciness',
    'danceability',
    'dyn_range_mean',
    'energy',
    'flatness',
    'instrumentalness',
    'key',
    'liveness',
    'loudness',
    'mechanism',
    'mode',
    'organism',
    'speechiness',
    'tempo',
    'time_signature',
    'valence',
    'acoustic_vector_0',
    'acoustic_vector_1',
    'acoustic_vector_2',
    'acoustic_vector_3',
    'acoustic_vector_4',
    'acoustic_vector_5',
    'acoustic_vector_6',
    'acoustic_vector_7']  # 73

ground_truth_index = train_col.index('skip_2') - 1  # -1 because we split the position field


def open_track_embedding(fname):
    track_features = pd.read_parquet(fname)
    return track_features[track_col].values


class AggSequence(tf.keras.utils.Sequence):
    """
    Aggregate results from multiple generator.
    Allows to avoid reading a single file (with a specific timestamp) and
    instead combine sevaral generators.
    """
    def __init__(self, generators):
        self.generators = generators

    def __len__(self):
        return len(self.generators[0])

    def on_epoch_end(self):
        for gen in self.generators:
            gen.on_epoch_end()

    def __getitem__(self, index):
        X_list = []
        Y_list = []
        for gen in self.generators:
            X_new, Y_new = gen[index]
            X_list.append(X_new)
            Y_list.append(Y_new)
        X_agg = []
        Y_agg = tf.concat(Y_list, axis=0)
        for k in range(len(X_list[0])):
            X_agg.append(tf.concat([x[k] for x in X_list], axis=0))
        return X_agg, Y_agg


class BatchSequence(tf.keras.utils.Sequence):
    """
    Input file generator that reads the parquet file and convert it to a
    keras.Sequence.

    That was the most efficient way I found to do this. But there may be better
    solutions to train on the spotify skip prediction dataset.

    Also the solution that allowed to avoid memory errors.
    """
    def __init__(self, filelist, batch_size = 500,
                sess_length = 20, centered_Y = False,
                tile_num = 1):
        self.filelist = filelist
        self.batch_size = batch_size
        self.sess_length = sess_length
        self.centered_Y = centered_Y
        self.tile_num = 1

        self.db_part = 0
        self.current_file = -1
        self.n_sess = 0
        self.indices = 0
        self.indices_slices = []

        self.train_data_song = 0
        self.train_data = 0

        self.on_epoch_end()


    def __len__(self):
        return self.n_sess // self.batch_size


    def __getitem__(self, index):
        ''' get a batch of data '''
        index = (index % self.__len__())  # because tf caches len only once
        X_left = self.train_data[index*self.batch_size:(index+1)*self.batch_size,:10, :]
        X_right = self.train_data[index*self.batch_size:(index+1)*self.batch_size,10:, :2]
        Y = self.train_data[index*self.batch_size:(index+1)*self.batch_size,10:,ground_truth_index:(ground_truth_index+1)]
        if self.centered_Y:
            Y = 2 * Y - 1  # [-1,1] instead of [0,1] for interpretable networks
        if self.tile_num > 1:
            Y = tf.tile(Y, (1, 1, self.tile_num))

        X_id_left = self.train_data_song[index*self.batch_size:(index+1)*self.batch_size,:10]
        X_id_right = self.train_data_song[index*self.batch_size:(index+1)*self.batch_size,10:]

        return (X_id_left, X_left, X_id_right, X_right), Y


    def on_epoch_end(self):
        ''' the annoying part: open a new file and manage log-sessions of variable lengths '''
        self.current_file = (self.current_file + 1) % len(self.filelist)
        filename = self.filelist[self.current_file]
        print('--- opening ', self.current_file, ':', filename)
        del self.db_part
        del self.train_data
        del self.train_data_song

        gc.collect()
        self.db_part = pd.read_parquet(filename)
        self.n_sess = len(self.db_part['session_id'].unique())

        self.train_data_song = np.zeros((self.n_sess * self.sess_length),
                                        dtype='int')   # song ids
        self.train_data = -2 * np.ones(
                    (self.n_sess * self.sess_length, len(train_col) - 1),
                    dtype='float32')   # song data, with -2. value for unavailable data

        # space(pandas w/ varlen sessions) -> space(sess_length * n_sess)
        a = self.db_part[['session_position', 'session_length']]
        offset = np.zeros(len(a), dtype='int')
        half_sel = a['session_position'] == ((a['session_length'] // 2) + 1)
        begin_sel = a['session_position'] == 1
        begin_sel.iloc[0] = False  # no offset on first el
        offset[half_sel] = self.sess_length // 2 - (a[half_sel]['session_length'] // 2)
        offset[begin_sel] = self.sess_length - offset[half_sel][:-1]
        self.indices = np.cumsum(offset) + a['session_position'] - 1  # cos' pos:[1..len]

        del a
        del offset
        del half_sel
        del begin_sel

        # fill numpy
        self.train_data_song[self.indices] = self.db_part[train_col[0]]
        self.train_data[self.indices] = self.db_part[train_col[1:]].values
        self.train_data_song = self.train_data_song.reshape(
                        (self.n_sess, self.sess_length, 1))
        self.train_data = self.train_data.reshape(
                        (self.n_sess, self.sess_length, len(train_col) - 1))


if not DEBUG:
    db_files = glob.glob(os.path.join(DATASET_PATH, '*.parquet'))  # 660 files
    train_files = db_files[:-100]
    valid_files = db_files[-100:-50]
    test_files = db_files[-50:]
    print('Found', len(train_files), 'files for training')
    print('Found', len(valid_files), 'files for validation')

    if AGG_GENERATORS:
        CUT = 4
        BS = 500  # batchsize to be cut in several pieces
        generators = []
        n_file = len(train_files) // CUT
        print('Cutting train data into', CUT, 'generators with', n_file, 'files each')
        for k in range(CUT):
            generators.append(BatchSequence(train_files[n_file*k:n_file*(k+1)],
                                centered_Y=True,
                                batch_size = BS // CUT))
        train_generator = AggSequence(generators)
    else:
        CUT = 1
        train_generator = BatchSequence(train_files, centered_Y=True)

    valid_generator = BatchSequence(valid_files, centered_Y=True)
    track_features = open_track_embedding(TRACK_F_PATH)
else:
    # mock generators for debugging, saves you muchos minutes of waiting
    CUT = 1
    train_generator = mock_gen(bs = 32)
    valid_generator = mock_gen(bs = 32)
    track_features = np.random.random((4, 29)) - 0.5  # 4 songs, has to be coherent with mock gen


##  INTERPRETABLE MODEL PARAMS

node_range_left = [
    (0, 2),  # sess pos
    (2, 15),  # context
    (15, 44),  # interaction
    (44, 47),  # track gen
    (47, 73),  # track mus
]

node_range_right = [
    (0, 2), # sess pos
    (2, 5), # track gen
    (5, 31), # track mus
]

S_left = [
    [1, 1, 0, 0, 0],  # context   -- no right
    [1, 0, 1, 0, 0],  # int       -- no right
    [1, 0, 0, 1, 0],  # track gen
    [1, 0, 0, 0, 1],  # track mus
    [1, 1, 1, 0, 0],  # log       -- no right
    [1, 0, 0, 1, 1],  # track
    [1, 1, 1, 1, 0],  # log + gen
    [1, 1, 1, 1, 1],  # all
    ]

S_right = [
    [1, 0, 0],  # blank
    [1, 1, 0],  # gen
    [1, 0, 1],  # mus
    [1, 1, 1],  # all
    ]

S_merge_left_right = [  # what should be included where between S_left and S_right
    [0, 0],  # left context <-> right context
    [1, 0],  # left interaction <-> right context (we don't have interactions)
    [2, 1],  # left track context <-> right track context
    [3, 2],  # left track music <-> right track music
    [4, 0],  # left all interactions <-> right context
    [5, 3],  # left all track data <-> right all track data
    [6, 1],  # ...
    [7, 3],  # ...
]

S_left = [[bool(x) for x in l] for l in S_left]
S_right = [[bool(x) for x in l] for l in S_right]

TR_left = generate_tr(S_left)  # Here, TR_ij = S_i \subset S_j !
tr_mat_left = tf.convert_to_tensor(TR_left, 'float32')
child_mat_left = tr_mat_left - tf.eye(tr_mat_left.shape[0])

TR_right = generate_tr(S_right)
tr_mat_right = tf.convert_to_tensor(TR_right, 'float32')
child_mat_right = tr_mat_right - tf.eye(tr_mat_right.shape[0])

params = {
    'sess_length': 20,
    'input_dim': 44,
    'track_embedding': track_features,
    'hidden_dim': 64,
    'heads': 8,  # for transformer models
    'interpretation': {
        'left': {
            'node_range': node_range_left,
            'S': S_left,
            'inclusion_mat': TR_left,
            'child_mat': child_mat_left,
        },
        'right': {
            'node_range': node_range_right,
            'S': S_right,
            'inclusion_mat': TR_right,
            'child_mat': child_mat_right,
        },
        'merge': S_merge_left_right,
    },
}


## INSTANTIATE MODEL

m = Int_Transformer(params)
del params['track_embedding']
del track_features
gc.collect()


## TRAIN

version = 0
model_name = m.save_name + '_' + str(int(time.time()))

if version > 0:
    m.model.load_weights('{}/{}_{}.h5'.format(SPOTIFY_WEIGHTS_PATH, model_name, version))
elif not DEBUG:
    np.save('{}/params_{}'.format(SPOTIFY_WEIGHTS_PATH, model_name), params)

saver = ModelCheckpoint('{}/{}_{}.h5'.format(SPOTIFY_WEIGHTS_PATH, model_name, version+1),
                        save_best_only=True,
                        save_weights_only=True,
                        )
lr_reduce = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.8,
                              patience=10,
                              verbose=1,
                              mode='auto', epsilon=0.0001,
                              cooldown=2,
                              min_lr=1e-7)

tensorboard_callback = TensorBoard(
        log_dir='{}/{}_{}'.format(SPOTIFY_LOG_PATH, model_name, version+1),
        histogram_freq=1)

training_callbacks = [lr_reduce, saver, tensorboard_callback]
if DEBUG:
    training_callbacks = []

m.model.fit(train_generator,
            epochs=1000, steps_per_epoch=300 * CUT, verbose=2,
            validation_data=valid_generator,
            validation_steps=200,
            callbacks=training_callbacks,
            workers=1,
            use_multiprocessing=False,
            )


## TEST

if not DEBUG:
    test_generator = BatchSequence(test_files, centered_Y=True)
else:
    test_generator = valid_generator

scores = []

for i in range(50):
    score = m.model_out.evaluate(
                test_generator,
                steps=50,
                workers=1,
                use_multiprocessing=False,
                verbose=2,
                )
    scores.append(score)

if not DEBUG:
    np.save('{}/{}_{}'.format(SPOTIFY_RESULTS_PATH, model_name, version), scores)