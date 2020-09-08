import random
from collections import defaultdict
import csv
import os
from tqdm import tqdm

import numpy as np
import tensorflow as tf

def swap(l, i, j):
    ''' Swoopy Swap '''
    l[i], l[j] = l[j], l[i]


class MovieLensDataset1M():
    def __init__(self, PATH, verbose=True, csv_delimiter='@', debug=False):
        '''
        Open from .csv
        '''
        if verbose: print('init');

        # mapping const for reproducibility
        genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        ages = [1, 18, 25, 35, 45, 50, 56]
        year_mean = 1986.1   # yey... i did precompute that, not super clean.
        year_std = 16.9      # used to normalise the `year` field


        # [uid]{ 'rates': <(int, float)>[`iid`, `rate`, `timestamp`]}
        ratings = defaultdict(lambda: { 'rates': [] })
        map_movies = []
        map_users = []
        with open(os.path.join(PATH, 'ratings.csv'), 'r', encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=csv_delimiter)
            if debug: i = 0
            for row in tqdm(reader):
                if debug:
                    i += 1
                    if i > 100000:
                        break

                user = int(row[0])
                item = int(row[1])
                try:
                    item_mapped = map_movies.index(item)
                except:
                    map_movies.append(item)
                    item_mapped = len(map_movies) - 1
                try:
                    user_mapped = map_users.index(user)
                except:
                    map_users.append(user)
                    user_mapped = len(map_users) - 1
                ratings[user_mapped]['rates'].append((item_mapped, float(row[2]), int(row[3]) ))
        if verbose: print('opened ratings.csv');

        # [iid]{ 'genres': <int>[`genre_index`] }
        # genres = <str>[]
        movies = defaultdict(lambda: { 'genres': [0] * len(genres), 'year': 0 })
        with open(os.path.join(PATH, 'movies.csv'), 'r', encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=csv_delimiter)
            for row in reader:
                item = int(row[0])
                try:
                    item_mapped = map_movies.index(item)
                except:
                    map_movies.append(item)
                    item_mapped = len(map_movies) - 1
                movies[item_mapped]['year'] = (float(row[1].split('(')[-1][:-1]) - year_mean) / year_std
                for movie_genre in row[2].split('|'):
                    movies[item_mapped]['genres'][genres.index(movie_genre)] = 1
        if verbose: print('read movies.csv');


        # [uid]{ 'genres': <int>[genre_index] }
        # genres = <str>[]
        with open(os.path.join(PATH, 'users.csv'), 'r', encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=csv_delimiter)
            for row in reader:
                user = int(row[0])
                try:
                    user_mapped = map_users.index(user)
                    ratings[user_mapped]['sex'] = int(row[1] == 'F')
                    ratings[user_mapped]['age'] = ages.index(int(row[2]))
                    ratings[user_mapped]['cat'] = int(row[3])
                except:
                    pass
        if verbose: print('read users.csv');


        self.Ni = len(map_movies)
        self.Nu = len(map_users)

        self.Ng = len(genres)

        self.Nc = 21  # max user_cat
        self.Na = len(ages)

        self.genres = genres
        self.map_movies = map_movies
        self.map_users = map_users

        self.ratings = ratings
        self.movies = movies


    def split_implicit_data(self):
        ''' Split train/test '''
        for user in self.ratings:
            self.ratings[user]['n'] = len(self.ratings[user]['rates'])

            # test_el is the last available timestamp
            test_el = max(self.ratings[user]['rates'], key=lambda x: x[2])
            test_el_index = self.ratings[user]['rates'].index(test_el)
            swap(self.ratings[user]['rates'], 0, test_el_index)
            self.ratings[user]['test_set'] = set([self.ratings[user]['rates'][0][0]])

            # val_el is a random rating element
            val_el_index = random.randint(1, self.ratings[user]['n'] - 1)
            swap(self.ratings[user]['rates'], 1, val_el_index)
            self.ratings[user]['val_set'] = set([self.ratings[user]['rates'][1][0]])

            self.ratings[user]['train_set'] = set([x[0] for x in self.ratings[user]['rates'][2:]])


    def get_implicit_data(self, D = 4, target_set = 'train_set'):
        ''' NeuCF-repo style where everything is computed in advance '''
        Xu = []
        Xi = []
        Y = []
        item_set = set(self.movies.keys())
        for user in tqdm(self.ratings):
            neg_item_set = item_set - self.ratings[user]['train_set'] # do not add val and test ! else implicitely learned

            #positive
            N = len(self.ratings[user][target_set])
            Xu += [user] * ((D + 1) * N)
            Y += ([1] + [0] * D) * N
            for pos_item in self.ratings[user][target_set]:
                Xi += [pos_item] + random.sample(neg_item_set, D)

        Xu = tf.convert_to_tensor(Xu, dtype='int32')
        Xi = tf.convert_to_tensor(Xi, dtype='int32')
        Y = tf.convert_to_tensor(Y, dtype='float32')
        return [Xu, Xi], Y


    def get_aug_implicit_data(self, D = 4, mode = 7, target_set = 'train_set'):
        '''
        Mode: binary value of what to add in input
            mode = has(CF_u) * 2^0 +  has(CF_i) * 2^1 +  has(CB) * 2^2
        It feels complicated but is actually easier to handle once you are
        used to the correspondence set <-> binary number.
        '''
        if mode & 1 > 0:
            Xu = []  # user
        if mode & 2 > 0:
            Xi = []  # item
        if mode & 4 > 0:
            Xs = []  # sex
            Xa = []  # age
            Xc = []  # cat
            Xm = []  # movie data

        Y = []
        item_set = set(self.movies.keys())
        for user in self.ratings:
            neg_item_set = item_set - self.ratings[user]['train_set']
            #positive
            N = len(self.ratings[user][target_set])
            if mode & 1 > 0:
                Xu += [user] * ((D + 1) * N)
            if mode & 4 > 0:
                Xs += [self.ratings[user]['sex']] * ((D + 1) * N)
                Xa += [self.ratings[user]['age']] * ((D + 1) * N)
                Xc += [self.ratings[user]['cat']] * ((D + 1) * N)

            Y += ([1] + [0] * D) * N
            for pos_item in self.ratings[user][target_set]:
                user_items = [pos_item] + random.sample(neg_item_set, D)
                if mode & 2 > 0:
                    Xi += user_items
                if mode & 4 > 0:
                    for it in user_items:
                        Xm.append([self.movies[it]['year']]
                                    + self.movies[it]['genres'])

        X = []
        if mode & 1 > 0:
            X.append(tf.convert_to_tensor(Xu, dtype='int32'))
        if mode & 2 > 0:
            X.append(tf.convert_to_tensor(Xi, dtype='int32'))
        if mode & 4 > 0:
            X.append(tf.convert_to_tensor(Xs, dtype='int32'))
            X.append(tf.convert_to_tensor(Xa, dtype='int32'))
            X.append(tf.convert_to_tensor(Xc, dtype='int32'))
            X.append(tf.convert_to_tensor(Xm, dtype='float32'))
        Y = tf.convert_to_tensor(Y, dtype='float32')
        return X, Y
