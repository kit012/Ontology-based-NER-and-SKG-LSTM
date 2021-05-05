import pandas as pd
from multiprocessing import Pool
from stock_prediction.util import config, Tokenizer, get_synonyms_and_phrases, find_nearest_trading_date, \
    check_list_identical
from stock_prediction.datasets import get_movement, Datasets
from itertools import chain
import statistics
from nltk import ngrams
import collections

from sklearn.metrics import classification_report, matthews_corrcoef


class OntologyNER(object):
    def __init__(self):
        self.oee = [
            'get', 'earn', 'profit', 'clean up', 'gross', 'take in', 'yield', 'make', 'realize', 'win back',
            'take', 'acquire', 'buy', 'purchase', 'subscribe', 'buy back', 'take over', 'receive', 'accept',
            'borrow', 'inherit', 'obtain', 'source', 'outsource', 'turn', 'raise', 'secure', 'shop', 'fundraise',
            'give', 'donate', 'contribute', 'deposit', 'sell', 'clear', 'trade', 'retail', 'market', 'pay',
            'bribe', 'spend', 'consume', 'waste', 'invest', 'fund', 'speculate'
        ]

        self.prep = [
            'in', 'on', 'at', 'under', 'for'
        ]

        self.indicator_words = self.oee + self.prep + [
            'be', 'predict',
            'have', 'will be', 'shall be', 'would be', 'may be', 'might be',
            'rise', 'drop', 'up', 'down', 'fall', 'loss', 'watch', 'gain', 'lead', 'great', 'beat', 'break',
            'accelerate', 'earn', 'hold', 'pay', 'hit', 'positive', 'negative',
            'oil prices',

        ]

        self.not_end_with = [
            'be', 'will be', 'shall be', 'would be', 'may be', 'might be',
            'in', 'on', 'at', 'under', 'for', 'to',
            '&', 'an', 'of'
        ]

        self.train_df = Datasets().get_training_set()

        synonyms, phrases = get_synonyms_and_phrases()
        self.tokenizer = Tokenizer(synonyms, phrases)

    @staticmethod
    def convert2lower(x):
        return ' '.join([i if (i.isupper() and i.isalpha()) else i.lower() for i in x['TITLE'].split(' ')])

    def end_with_indicator_words(self, x):
        for i in self.not_end_with:
            if x.split(' ')[-1] == i:
                return True
        return False

    def check_indicator_words(self, x):
        for i in self.indicator_words:
            for j in x:
                if i == j:
                    return True
        return False

    def check_prep(self, x):
        for i in self.prep:
            for j in x:
                if i == j:
                    return True
        return False

    def get_ngrams(self, x, n=4):
        res = []
        for n in range(1, n + 1):
            #     print(list(ngrams(x, n)))
            grams = []
            for i in ngrams(x, n):
                if '<SEP>' not in i and '/' not in i and self.check_indicator_words(i):
                    word = ' '.join(i)
                    if not word.startswith('-') and not word.startswith('&') and not word.endswith(
                            '-') and not self.end_with_indicator_words(word):
                        grams.append(word)
            res.append(grams)
        return [w for w in set(chain.from_iterable(res)) if w not in self.indicator_words]

    def preprocess(self):
        self.train_df['TAG'] = self.train_df['TAG'].apply(lambda x: str(x).split(','))
        self.train_df = self.train_df.explode('TAG')
        self.train_df['TITLE'] = self.train_df.swifter.apply(self.convert2lower, axis=1)

        print(self.train_df.count())

        tokenized_title = []
        threads = 24

        titles = self.train_df['TITLE'].tolist()

        with Pool(threads) as p:
            tokenized_title.append(p.map(self.tokenizer.tokenize, titles))

        train_tokenized_title = pd.DataFrame(tokenized_title[0], columns=['tokenized_title', 'ner'])

        self.train_df = pd.concat([self.train_df.reset_index(), train_tokenized_title], axis=1)
        movement = get_movement()

        self.train_df['DTT'] = self.train_df['DT'].apply(find_nearest_trading_date)
        self.train_df = pd.merge(self.train_df, movement, left_on='DTT', right_on='Date', how='left').drop(
            columns=['Date'])

    def find_keywords(self, titles):
        # ngram
        ngrams = []
        for t in titles:
            ngrams.append(self.get_ngrams(t))

        keywords = collections.Counter(chain.from_iterable(ngrams))
        #     mean_for_gt_2 = statistics.mean([itm for key, itm in keywords.items() if itm > 2])
        mean_for_gt_2 = statistics.median([itm for key, itm in keywords.items()])

        # filter the keywords that count over the mean greater than 2
        new_keys = []

        for k in keywords:
            if keywords.get(k) > mean_for_gt_2:
                new_keys.append(k)

        ner = {k: keywords[k] for k in new_keys}

        # filter the NER which is subset of others
        cleanned_ners = []

        for k1, itm1 in ner.items():
            #         subset_of_others = False
            #         for k2, itm2 in ner.items():
            #             if k1 == k2:
            #                 continue
            #             elif k1 in k2:
            #                 subset_of_others = True
            #                 break
            #         if not subset_of_others:
            cleanned_ners.append(k1)
        return set(cleanned_ners)

    def get_ontology_ner(self, df, stock_code):
        pos = df[(df['TAG'] == stock_code) & (df[stock_code + '_FLAG'] == 1)]['tokenized_title'].tolist()
        neg = df[(df['TAG'] == stock_code) & (df[stock_code + '_FLAG'] == 0)]['tokenized_title'].tolist()
        # print(df[(df['TAG'] == stock_code) & (df[stock_code + '_FLAG'] == 1)]['tokenized_title'].count())
        # print(df[(df['TAG'] == stock_code) & (df[stock_code + '_FLAG'] == 0)]['tokenized_title'].count())

        pos_ner = self.find_keywords(pos)
        neg_ner = self.find_keywords(neg)

        # filter the noise
        return [pos_ner - neg_ner, neg_ner - pos_ner]

    @staticmethod
    def match_keywords(title, pos, neg):
        p_score, n_score = 0, 0
        for p in pos:
            if p in title:
                p_score += 1
        for n in neg:
            if n in title:
                n_score -= 1
        return p_score, n_score

    def prediction_movement(self, titles, pos, neg):
        p_score, n_score = 0, 0

        for t in titles:
            sent, ner = self.tokenizer.tokenize(t)
            ps, ns = self.match_keywords(' '.join(sent), pos, neg)
            p_score += ps
            n_score -= ns
        return [p_score, n_score]

    def evaluate(self):
        for s in config.stock_list:

            pos, neg = self.get_ontology_ner(self.train_df, s)
            # print(pos, neg)

            test_df = pd.read_parquet('out/testing_set/testing_news_2d_' + s + '.parquet')
            test_df['PRED'] = test_df['TITLE'].swifter.apply(lambda x: self.prediction_movement(x, pos, neg))
            test_df['PRED_MOVEMENT'] = test_df['PRED'].apply(lambda x: 0 if x[0] < x[1] else 1)
            test_df['MOVEMENT'] = test_df[s].apply(lambda x: 0 if x < 0 else 1)

            idx = []
            for index, row in test_df.iterrows():
                if not check_list_identical(row['PRED'], [0, 0]):
                    idx.append(index)

            print(s)
            print(classification_report(test_df['MOVEMENT'], test_df['PRED_MOVEMENT']))
            print('MCC: ', matthews_corrcoef(test_df['MOVEMENT'], test_df['PRED_MOVEMENT']))
