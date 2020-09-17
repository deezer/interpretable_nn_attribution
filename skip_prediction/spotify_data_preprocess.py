"""
Spotify data CSV dataset file processing and converter using pandas
"""

import glob
import pickle
import os
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PATH_DATA = 'spotify_data'
PATH_EXPORT_LOG = 'spotify_parqueted'
PATH_EXPORT_TRACK = 'data'
ENCODE_LABELS = True


## open track data

track_files = glob.glob(PATH_DATA + '/track_features/' + '*.csv')
train_files = glob.glob(PATH_DATA + '/training_set/' + '*.csv')


## normalise track data fields

def save_trackid(fname, le):
    f = open(fname, 'wb')
    pickle.dump(le, f)
    f.close()

def open_trackid(fname):
    f = open(fname, 'rb')
    le = pickle.load(f)
    f.close()
    return le


track_tables = []
for fname in tqdm(track_files):
    a = pd.read_csv(fname)
    track_tables.append(a)
track_data = pd.concat(track_tables)
track_data['mode'] = (track_data['mode'] == 'major').map(float)

TRACK_COL = ['duration', 'release_year', 'us_popularity_estimate',
       'acousticness', 'beat_strength', 'bounciness', 'danceability',
       'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mechanism', 'mode', 'organism', 'speechiness',
       'tempo', 'time_signature', 'valence', 'acoustic_vector_0',
       'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3',
       'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6',
       'acoustic_vector_7']

TRACK_NORM = {
    'mean': {},
    'std': {},
}

for key in tqdm(TRACK_COL):
    TRACK_NORM['mean'][key] = track_data[key].mean()
    TRACK_NORM['std'][key] = track_data[key].std()
    track_data[key] = (track_data[key] - TRACK_NORM['mean'][key]) / TRACK_NORM['std'][key]

if ENCODE_LABELS:
    le = LabelEncoder()
    le.fit(track_data['track_id'])
    save_trackid(os.path.join(PATH_EXPORT_TRACK, 'le_trackid.pkl'), le)
else:
    le = open_trackid(os.path.join(PATH_EXPORT_TRACK, 'le_trackid.pkl'))


## normalise log data and save to parquet

SESSION_ONEHOT = {
    "context_type": ['radio', 'catalog', 'personalized_playlist', 'editorial_playlist', 'user_collection', 'charts'],
    "hist_user_behavior_reason_start": ['endplay', 'fwdbtn', 'clickrow', 'trackdone', 'uriopen', 'remote', 'playbtn', 'backbtn', 'trackerror', 'popup', 'appload'],
    "hist_user_behavior_reason_end": ['endplay', 'fwdbtn', 'clickrow', 'trackdone', 'uriopen', 'remote', 'playbtn', 'backbtn', 'trackerror', 'popup', 'appload'],
}


for train_file in tqdm(train_files):
    session_data = pd.read_csv(train_file)  # ~10s

    session_data['hour_of_day'] = session_data['hour_of_day'] / 24
    session_data['date_gap'] = pd.to_datetime(session_data['date']) - pd.Timestamp(2018, 7, 13)
    session_data['date_gap'] = session_data['date_gap'].dt.days

    for field in SESSION_ONEHOT:
        for sub_field in SESSION_ONEHOT[field]:
            session_data.insert(0, field + '_' + sub_field, session_data[field] == sub_field)

    del session_data['context_type']
    del session_data['hist_user_behavior_reason_start']
    del session_data['hist_user_behavior_reason_end']
    del session_data['date']

    session_data.insert(0, 'ordinal_track_id_clean',
        le.transform(session_data['track_id_clean']))

    session_data.to_parquet(PATH_EXPORT_LOG + '/' + train_file.split('/')[-1] + '.parquet')
