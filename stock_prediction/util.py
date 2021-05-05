import collections
from collections import namedtuple
import numpy as np
from datetime import datetime, timedelta
import spacy
from nltk.stem import WordNetLemmatizer

from stock_prediction.config import Config

import pickle5 as pickle

nlp = spacy.load("en_core_web_lg")
lemmatizer = WordNetLemmatizer()
config = Config()


def get_synonyms_and_phrases():
    with open('data/synonyms.pkl', 'rb') as f:
        synonyms = pickle.load(f)

    phrases = []
    with open('data/phrases.txt') as f:
        for line in f:
            phrases.append(line)

    return synonyms, phrases


def accumulate_list(num_list):
    return [sum(num_list[:y]) for y in range(1, len(num_list) + 1)]


def add_items_to_dict(d, key, item):
    if key in d:
        items = d[key]
        if item not in items:
            items.append(item)
            d[key] = items
    else:
        d[key] = [item]


def check_list_identical(list1, list2):
    if collections.Counter(list1) == collections.Counter(list2):
        return True
    else:
        return False


def combine_rel(r1, r2):
    for k in r2.keys():
        rel = r1.get(k)
        if rel is not None:
            r1[k] = r1[k] + 1
        else:
            r1[k] = 1
    return r1


def find_key_by_value_list(dictionary, value_list):
    for k, v in dictionary.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if check_list_identical(v, value_list):
            return k


def find_nearest_trading_date(d, win_size=2):
    if isinstance(d, str):
        d = datetime.strptime(d, '%Y-%m-%d').date()
    elif isinstance(d, datetime):
        d = d.date()

    for i in range(0, 10):
        tmp_date = (d + timedelta(days=i + win_size)).strftime('%Y-%m-%d')
        if tmp_date in config.trading_date_list:
            return tmp_date
    return None


def k_wins_all(x):
    c = dict(collections.Counter(x))
    up = c.get(1)
    down = c.get(0)

    if up is not None and down is not None:
        if up > down:
            return 1
        else:
            return 0
    elif up is None:
        return 0
    else:
        return 1


def get_items_in_dict(d, key):
    if isinstance(key, str):
        return d.get(key)
    elif isinstance(key, list):
        itms = []
        for k in key:
            if d.get(k) is not None:
                itms += d.get(k)
        if len(itms) != 0:
            return itms


def match_keywords(title, pos, neg):
    p_score, n_score = 0, 0
    for p in pos:
        if p in title:
            p_score += 1
    for n in neg:
        if n in title:
            n_score -= 1
    return p_score, n_score


def prediction_movement(titles, pos, neg):
    p_score, n_score = 0, 0
    synonyms, phrases = get_synonyms_and_phrases()
    tokenizer = Tokenizer(synonyms, phrases)

    for t in titles:
        sent, ner = tokenizer.tokenize(t)
        ps, ns = match_keywords(' '.join(sent), pos, neg)
        p_score += ps
        n_score -= ns
    return [p_score, n_score]


def replace_synonyms_words(x):
    synonyms, _ = get_synonyms_and_phrases()

    for v in synonyms.values():
        for s in v:
            if s.lower() in x.lower():
                start_idx = x.lower().index(s.lower())
                s = x[start_idx: start_idx + len(s)]
                x = x.replace(s, find_key_by_value_list(synonyms, v))
    return x


def use_post_date_if_available(up_dt, post_dt):
    the_date = post_dt if post_dt is not np.nan else up_dt
    return datetime.strptime(the_date, '%Y-%m-%d %H:%M:%S')


class Tokenizer(object):
    def __init__(self, synonyms, phrases):
        self.synonyms = synonyms
        self.phrases = phrases

        self.stopwords = [
            'he', 'she', 'we', 'they', 'it', 'its', 'own', 'you', 'your', 'him', 'his', 'her', 'this', 'those', 'these'
        ]

    def slice_with_phrase(self, word):
        tokenized_word = []

        for t in word:
            for ph in self.phrases:
                if ph in t:
                    start_pos = t.index(ph)
                    end_pos = t.index(ph) + len(ph)
                    if t[:start_pos] != '':
                        tokenized_word = self.slice_with_phrase([t[:start_pos]]) + tokenized_word
                    elif t[:start_pos] != '':
                        tokenized_word.append(t[:start_pos])

                    tokenized_word.append(t[start_pos:end_pos])

                    if t[:start_pos] != '':
                        tokenized_word = tokenized_word + self.slice_with_phrase([t[end_pos:]])
                    elif t[end_pos:] != '':
                        tokenized_word.append(t[end_pos:])

                    break
        if len(tokenized_word) == 0:
            return word
        else:
            return tokenized_word

    def mark_phrase_flag(self, tokenized_title):
        phrase_flag = []
        for t in tokenized_title:
            if t in self.phrases:
                phrase_flag.append(True)
            else:
                phrase_flag.append(False)
        return phrase_flag

    @staticmethod
    def get_word_from_token(token, word):
        return word[token['start']:token['end']]

    def tokenized_by_spacy(self, title, tokens, ner):
        sent = []
        skip_count = 0

        for idx, d in enumerate(tokens):
            #         print(idx, d['lemma'], d['pos'], d['dep'])
            if skip_count == 0:
                if title[d['start']:d['end']] in [',', '.', ':', ';', '(', ')']:
                    sent.append('<SEP>')

                elif d['pos'] == 'PUNCT' and d['lemma'] != '-':
                    continue
                # For 's and a
                elif d['lemma'] in ["'s", "’s", "’s", "n't", "n’t", "'re", 'a', 'an', '’', "'", 'the'] + self.stopwords:
                    continue
                # For proper noun
                elif d['pos'] == 'PROPN':
                    word = title[d['start']:d['end']]
                    sent.append(word.lower())
                # For Money
                elif d['tag'] == '$':
                    word = '<$>'
                    for i in range(idx + 1, len(tokens)):
                        if tokens[i]['tag'] == 'CD':
                            skip_count += 1
                        else:
                            break
                    sent.append(word)
                # For NUM
                elif d['pos'] == 'NUM':
                    if d['lemma'] in ner.get('DATE', ''):
                        sent.append(d['lemma'])
                    else:
                        word = '<NUM>'
                        if idx != len(tokens) - 1 and tokens[idx + 1]['pos'] != 'ADP':
                            word += " " + title[tokens[idx + 1]['start']:tokens[idx + 1]['end']]
                            skip_count += 1
                        sent.append(word)

                elif d['lemma'] != '-PRON-':
                    sent.append(d['lemma'])
            else:
                skip_count = skip_count - 1
        return sent

    def tokenize(self, title):
        title = replace_synonyms_words(title)

        doc = nlp(title)

        ner = {}
        tokens = doc.to_json()['tokens']
        ents = doc.to_json()['ents']

        for e in ents:
            add_items_to_dict(ner, e['label'], self.get_word_from_token(e, title))

        t = [i for i in tokens if i['start'] >= 0 and i['end'] <= len(title)]
        sent = self.tokenized_by_spacy(title, t, ner)

        return [sent, ner]


class Relation(object):
    def __init__(self, title):
        self.title = title

        doc = nlp(self.title)
        self.tokens = doc.to_json()['tokens']

    def find_dep_id(self, token_id):
        dep = {}
        for i in self.tokens:
            if i['head'] == token_id and i['id'] != token_id:
                add_items_to_dict(dep, i['dep'], i['id'])
        return dep

    def get_word_from_token(self, token):
        return self.title[token['start']:token['end']]

    def get_subj_from_dep_verb(self, verb_id):
        subj = self.find_dep_id(verb_id).get('nsubj')
        if subj is not None:
            return self.get_phrase(subj[0], 'noun')

    def get_obj_from_dep_verb(self, verb_id, verb_type='verb'):
        prep = self.find_dep_id(verb_id).get('prep')
        if prep is not None:
            verb_id = self.get_phrase(verb_id, 'verb', return_last_id=True)

        if verb_type == 'verb':
            obj = get_items_in_dict(self.find_dep_id(verb_id), ['pobj', 'dobj', 'npadvmod', 'advmod'])
            if obj is not None:
                return self.get_phrase(obj[0], 'noun')

        elif verb_type == 'aux':
            obj = get_items_in_dict(self.find_dep_id(verb_id), ['acomp', 'pobj'])
            if obj is not None:
                return self.get_phrase(obj[0], 'aux')

    def find_dep(self, token_id, dep_type):
        deps = []
        if dep_type == 'acomp':
            dep_ids = self.find_dep_id(token_id).get('advmod')
            if dep_ids is not None:
                deps += dep_ids
        else:
            dep_ids = self.find_dep_id(token_id).get(dep_type)
            if dep_ids is not None:
                deps += dep_ids
                for i in dep_ids:
                    if len(self.find_dep(i, dep_type)) != 0:
                        deps += self.find_dep(i, dep_type)
        deps.sort()
        return deps

    def get_phrase(self, token_id, phrase_type, return_last_id=False):
        phrase = self.get_word_from_token(self.tokens[token_id])

        if phrase_type == 'noun':
            for j in ['compound', 'nummod', 'quantmod', 'npadvmod', 'poss', 'prep', 'pobj']:
                for i in self.find_dep(token_id, j):
                    phrase = self.get_word_from_token(self.tokens[i]) + ' ' + phrase
            last_id = token_id

        elif phrase_type == 'verb':
            for i in self.find_dep(token_id, 'prep'):
                phrase += ' ' + self.get_word_from_token(self.tokens[i])
            for i in self.find_dep(token_id, 'prt'):
                phrase += ' ' + self.get_word_from_token(self.tokens[i])

            if len(self.find_dep(token_id, 'prep')) != 0:
                last_id = self.find_dep(token_id, 'prep')[0]
            else:
                last_id = token_id

        elif phrase_type == 'aux':
            for j in ['acomp']:
                for i in self.find_dep(token_id, j):
                    phrase = self.get_word_from_token(self.tokens[i]) + ' ' + phrase

            for j in ['prep', 'prt']:
                for i in self.find_dep(token_id, j):
                    phrase += ' ' + self.get_word_from_token(self.tokens[i])

            if len(self.find_dep(token_id, 'acomp')) != 0:
                last_id = self.find_dep(token_id, 'acomp')[0]
            else:
                last_id = token_id

        if return_last_id:
            return last_id
        else:
            edge = phrase.split(' ')
            return ' '.join([edge[i] for i in np.array([self.title.index(i) for i in edge]).argsort()])

    def find_relation(self):
        #         print(self.tokens)
        rel = []

        for idx, d in enumerate(self.tokens):

            if d['pos'] == 'VERB':
                obj = self.get_obj_from_dep_verb(d['id'])
                if obj is None:
                    obj = self.get_parataxis_obj(d['id'])

                rel.append(
                    {
                        'subject': self.get_subj_from_dep_verb(d['id']),
                        'verb': lemmatizer.lemmatize(self.get_phrase(d['id'], 'verb').lower()),
                        'object': obj
                    }
                )
            elif d['tag'] in ('VBP', 'VBN', 'VBZ'):
                rel.append(
                    {
                        'subject': self.get_subj_from_dep_verb(d['id']),
                        'verb': lemmatizer.lemmatize(self.get_phrase(d['id'], 'aux').lower()),
                        'object': self.get_obj_from_dep_verb(d['id'], verb_type='aux')
                    }
                )
        return [r for r in rel if r['subject'] is not None or r['object'] is not None]

    def get_parataxis_obj(self, verb_id):
        if self.tokens[verb_id]['dep'] in ['parataxis']:
            return self.get_subj_from_dep_verb(self.tokens[verb_id]['head'])


class VerbRelation(object):
    def __init__(self, title, dt=None):
        self.title = title
        self.dt = dt

        doc = nlp(self.title)
        self.tokens = doc.to_json()['tokens']
        self.relation = {}
        self.Rel = namedtuple("Relation", ["node1", "edge", "node2"])
        self.Rel_dt = namedtuple("Relation", ["node1", "edge", "node2", "date"])

        self.stopwords = [
            '%', 'a', "'s", "®", '&', "'", '-', '’s', '™'
        ]

    @staticmethod
    def preprocess(x):
        x = x.lower()
        x = lemmatizer.lemmatize(x)

        return x

    def is_stopwords(self, x):
        if x not in self.stopwords:
            return False
        else:
            return True

    def get_word_rel(self):
        for i in range(len(self.tokens)):
            n1 = self.get_word_from_token(self.tokens[i])
            dep = self.find_dep_id(i)

            #             print(self.tokens[i]['tag'])

            if len(dep) != 0:
                for k in dep.keys():
                    for n2_idx in dep[k]:
                        node1 = self.preprocess(n1)
                        node2 = self.preprocess(self.get_word_from_token(self.tokens[n2_idx]))

                        if not (node1.isspace() or node2.isspace() or k == 'punct' or self.is_stopwords(
                                node1) or self.is_stopwords(node2)):
                            r = self.Rel(node1=node1, edge=k, node2=node2)

                            if self.relation.get(r) is None:
                                self.relation[r] = 1
                            else:
                                self.relation[r] = self.relation.get(r) + 1

        return self.relation

    def get_word_rel_dt(self):
        for i in range(len(self.tokens)):
            n1 = self.get_word_from_token(self.tokens[i])
            dep = self.find_dep_id(i)

            if len(dep) != 0:
                for k in dep.keys():
                    for n2_idx in dep[k]:
                        node1 = self.preprocess(n1)
                        node2 = self.preprocess(self.get_word_from_token(self.tokens[n2_idx]))

                        if not (node1.isspace() or node2.isspace() or k == 'punct' or self.is_stopwords(
                                node1) or self.is_stopwords(node2)):
                            r = self.Rel_dt(node1=node1, edge=k, node2=node2, date=self.dt)

                            if self.relation.get(r) is None:
                                self.relation[r] = 1
                            else:
                                self.relation[r] = self.relation.get(r) + 1
        return self.relation

    def get_word_from_token(self, token):
        return self.title[token['start']:token['end']]

    def find_dep_id(self, token_id):
        dep = {}
        for i in self.tokens:
            if i['head'] == token_id and i['id'] != token_id:
                add_items_to_dict(dep, i['dep'], i['id'])
        return dep
