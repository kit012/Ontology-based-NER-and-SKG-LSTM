import torch
from torch import cuda
from torch.optim import Adam

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.models import TorusEModel
from torchkge.data_structures import KnowledgeGraph

from tqdm.autonotebook import tqdm

import pandas as pd
import numpy as np
from stock_prediction.model.word2vec import Word2vec
from stock_prediction.util import config, combine_rel, replace_synonyms_words, VerbRelation, find_nearest_trading_date
from stock_prediction.datasets import Datasets, get_movement


class KnowledgeGraphEmbedding(object):
    def __init__(self, stock_code):
        self.stock_code = stock_code

        self.emb_dim = config.kge_emb_dim
        self.lr = config.kge_lr
        self.n_epochs = config.kge_n_epochs
        self.b_size = config.kge_b_size
        self.margin = config.kge_margin

        self.train_df = Datasets().get_training_set()
        self.X_word, self.valid_class, self.labels = Word2vec().get_word2vec_model()

        self.final_rel = {}
        self.kg_train = None
        self.model = None

    def prepare_kg_train(self):
        self.final_rel = {}

        for x in self.train_df[self.train_df.TAG == self.stock_code]['TITLE'].tolist():
            x = replace_synonyms_words(x)
            # x = x.replace('\xa0', '')

            rel = VerbRelation(x).get_word_rel()
            self.final_rel = combine_rel(self.final_rel, rel)

        df_kg_train = pd.DataFrame([
            {
                'from': i[0].node1,
                'to': i[0].node2,
                'rel': i[0].edge

            }
            for i in self.final_rel.items()])

        self.kg_train = KnowledgeGraph(df_kg_train)

    def forward(self):
        # Define the model and criterion
        model = TorusEModel(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel, dissimilarity_type='torus_L2')
        criterion = MarginLoss(self.margin)

        # Move everything to CUDA if available
        if cuda.is_available():
            cuda.empty_cache()
            model.cuda()
            criterion.cuda()

        # Define the torch optimizer to be used
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        sampler = BernoulliNegativeSampler(self.kg_train)
        dataloader = DataLoader(self.kg_train, batch_size=self.b_size, use_cuda='all')

        iterator = tqdm(range(self.n_epochs), unit='epoch')
        for epoch in iterator:
            running_loss = 0.0
            for i, batch in enumerate(dataloader):
                h, t, r = batch[0], batch[1], batch[2]
                n_h, n_t = sampler.corrupt_batch(h, t, r)

                optimizer.zero_grad()

                # forward + backward + optimize
                pos, neg = model(h, t, n_h, n_t, r)
                loss = criterion(pos, neg)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                                      running_loss / len(dataloader)))

        model.normalize_parameters()
        model.ent_emb.cpu()
        return model

    def ent2vector(self, x):
        try:
            return self.model.ent_emb(torch.tensor([self.kg_train.ent2ix[x]], dtype=torch.long)).detach().numpy()[0]
        except:
            return None

    def embedding(self, x):
        vec = np.asarray([i for i in map(self.ent2vector, x) if i is not None])
        return np.concatenate((vec.min(axis=0), vec.max(axis=0)))

    def get_embedding_matrix(self):
        self.prepare_kg_train()
        self.model = self.forward()

        # nodes = list(
        #     set([i[0].node1 for i in self.final_rel.items()]).union(set([i[0].node2 for i in self.final_rel.items()])))
        # nodes.sort()
        #
        # df_node = pd.DataFrame(nodes, columns=['node'])
        # df_node.reset_index(inplace=True)
        #
        # df_node['emb'] = df_node['node'].apply(
        #     lambda x: self.model.ent_emb(torch.tensor([self.kg_train.ent2ix[x]], dtype=torch.long)).detach().numpy()[0])
        # df_node['emb'] = df_node['emb'].apply(lambda x: x.astype(np.float32))
        #
        # for i in self.valid_class:
        #     df_group = df_node[df_node.node.isin(self.X_word[np.argwhere(self.labels == i)].reshape(-1))]
        #     if df_group.shape[0] > 1:
        #         df_new = pd.DataFrame([[df_node['index'].max() + 1, ','.join(df_group['node'].tolist()),
        #                                 np.average(np.array(df_group['emb'].tolist()), axis=0)]],
        #                               columns=['index', 'node', 'emb'])
        #         df_node = df_node.append(df_new)
        #         df_node = df_node.drop(df_group.index)
        #
        # df_node.reset_index(drop=True, inplace=True)
        # df_node['node'] = df_node['node'].apply(lambda x: x.split(','))
        # df_node = df_node.explode('node')

        self.train_df.reset_index(inplace=True)
        self.train_df['DT'] = self.train_df['DT'].swifter.apply(find_nearest_trading_date)
        self.train_df = self.train_df[self.train_df.TAG == self.stock_code].groupby('DT')['TITLE'].apply(list)
        self.train_df = self.train_df.reset_index()

        train_rel = {}

        for index, row in self.train_df.iterrows():
            for x in row['TITLE']:
                x = replace_synonyms_words(x)
                x = x.replace('\xa0', '')

                rel = VerbRelation(x, row['DT']).get_word_rel_dt()
                train_rel = combine_rel(train_rel, rel)

        node1 = pd.DataFrame([{'node': i[0].node1, 'date': i[0].date} for i in train_rel.items()])
        node2 = pd.DataFrame([{'node': i[0].node2, 'date': i[0].date} for i in train_rel.items()])

        df_node_train = node1.append(node2)
        df_node_train.drop_duplicates(inplace=True)
        df_node_train = df_node_train.groupby('date')['node'].apply(list).reset_index()

        df_node_train['emb'] = df_node_train['node'].apply(self.embedding)

        df_node_train = pd.merge(df_node_train, get_movement(), left_on='date', right_on='Date')

        return df_node_train
