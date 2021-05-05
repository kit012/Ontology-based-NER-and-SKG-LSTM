import tensorflow as tf
from focal_loss import BinaryFocalLoss
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, matthews_corrcoef

from stock_prediction.util import replace_synonyms_words


class SKGLSTMModel(object):
    def __init__(self, df_node_train, stock_code, grouping=True):
        self.stop_words = set(stopwords.words('english'))
        self.stock_code = stock_code
        self.grouping = grouping
        self.df_node_train = df_node_train
        self.X_train = np.array(df_node_train['emb'].tolist())
        self.y = df_node_train[stock_code + '_FLAG']

        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))

    def forward(self):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

        model_classify = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=True,
                                 input_shape=(self.X_train.shape[0], self.X_train.shape[2])),
            tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(50, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1)
        ])

        model_classify.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                               optimizer=tf.optimizers.Nadam(learning_rate=0.001),
                               metrics=['mae', 'acc'])
        # model_classify.compile(loss=BinaryFocalLoss(gamma=2), optimizer=tf.optimizers.Nadam(learning_rate=0.001),
        # metrics=['mae', 'acc'])

        model_classify.summary()

        history = model_classify.fit(self.X_train, self.y, batch_size=20, epochs=100, callbacks=[callback],
                                     shuffle=True)

        model_classify.save('out/skg_lstm_model/' + self.stock_code + '.h5')

        return model_classify, history

    def preprocess(self, i):
        out = nltk.word_tokenize(i)
        out = [x.lower() for x in out]
        out = [x for x in out if x not in self.stop_words]
        return out

    def tokenize(self, titles):
        result = []
        for x in titles:
            x = replace_synonyms_words(x)
            x = x.replace('\xa0', '')
            x = self.preprocess(x)
            result += x
        return result

    def ent2vector(self, x):
        try:
            return self.df_node_train[self.df_node_train.node == x].iloc[0, 2]
        except:
            return None

    def embedding(self, x):
        vec = np.asarray([i for i in map(self.ent2vector, x) if i is not None])
        return np.concatenate((vec.min(axis=0), vec.max(axis=0)))

    def evaluate(self):
        model_classify, _ = self.forward()

        df_test = pd.read_parquet('out/testing_set/testing_news_2d_' + self.stock_code + '.parquet')
        df_test = df_test[['DT', 'TITLE', self.stock_code + '_FLAG']]
        df_test['emb'] = df_test['TITLE'].apply(lambda x: self.embedding(self.tokenize(x)))

        X_test = np.array(df_test['emb'].tolist())
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        print(classification_report(df_test[self.stock_code + '_FLAG'].values,
                                    [0 if i < 0.5 else 1 for i in model_classify.predict(X_test).reshape(-1)]))

        print('MCC: ', matthews_corrcoef(df_test[self.stock_code + '_FLAG'].values,
                                         [0 if i < 0.5 else 1 for i in model_classify.predict(X_test).reshape(-1)]))
