import numpy as np

import gensim.models

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import re

from multiprocessing import Pool
from collections import Counter

from stock_prediction.datasets import Datasets
from stock_prediction.util import config


class Word2vec(object):
    def __init__(self):
        self.dataset = Datasets()

        self.train_df = self.dataset.get_training_set()

        self.threads = config.threads
        self.cut_off_date = config.cut_off_date
        self.window_size = config.window_size
        self.train_word2vec = False
        self.max_group_num = config.word2vec_max_group_num
        self.similarity_method = config.word2vec_similarity_method
        self.eps = config.word2vec_eps
        self.min_samples = config.word2vec_min_samples

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, sent):
        out = nltk.word_tokenize(sent)
        out = [x.lower() for x in out]
        out = [x for x in out if x not in self.stop_words]
        return out

    @staticmethod
    def plot(X, labels, core_samples_mask, noise_class, n_clusters_):
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k in noise_class:
                continue

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()

    def get_word2vec_model(self):

        if self.train_word2vec is True:
            print('Train word2vec model')
            df_yf = self.dataset.get_yahoo_finance(drop_duplicates_only=True)
            df_kaggle = self.dataset.get_kaggle_benzinga_data_set()

            sentences = []
            titles = df_yf['TITLE'].tolist() + df_kaggle['title'].tolist()

            with Pool(self.threads) as p:
                sentences.append(p.map(self.preprocess, titles))

            word2vec_model = gensim.models.Word2Vec(sentences=sentences[0], sg=1)
            word2vec_model.save("out/word2vec.model")

        else:
            word2vec_model = gensim.models.Word2Vec.load('out/word2vec.model')

        unigram = []
        # remove punctuation
        p = re.compile(r'[^\w\s]')
        # remove number only
        p2 = re.compile(r'^[0-9]+$')
        # remove number
        p3_1 = re.compile(r'^[0-9]+([a-z]+|\+|-)?$')
        p3_2 = re.compile(r'^[0-9]+(\.|,|-|/)?[0-9a-z]+([a-z]+)?$')
        # remove date and time
        p4_1 = re.compile(r'^[0-9]+/[0-9]+/[0-9]+$')
        p4_2 = re.compile(r'^[0-9]+:[0-9]+(am|pm)$')

        for i in self.train_df.TITLE.tolist():
            words = nltk.word_tokenize(i)
            for w in words:
                w = w.lower()
                if w not in self.stop_words and not p.match(w) and not p2.match(w) and not p3_1.match(
                        w) and not p3_2.match(
                    w) and not p4_1.match(w) and not p4_2.match(w):
                    unigram.append(self.lemmatizer.lemmatize(w))

        unigram = list(set(unigram))
        unigram.sort()
        print("Number of unique words:", len(unigram))

        def vectorize(word):
            try:
                return word2vec_model.wv[word]
            except KeyError:
                return None

        unigram_dict = dict(zip(unigram, map(vectorize, unigram)))
        X = np.array([i[1] for i in unigram_dict.items() if i[1] is not None])
        X_word = np.array([i[0] for i in unigram_dict.items() if i[1] is not None])

        db = DBSCAN(metric=self.similarity_method, eps=self.eps, min_samples=self.min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        noise_class = [i[0] for i in Counter(labels).items() if i[1] > self.max_group_num]
        valid_class = [i[0] for i in Counter(labels).items() if i[1] <= self.max_group_num]

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        for i in range(-1, n_clusters_):
            print(i, ': ', X_word[np.argwhere(labels == i).reshape(-1)])

        # self.plot(X, labels, core_samples_mask, noise_class, n_clusters_)

        return X_word, valid_class, labels
