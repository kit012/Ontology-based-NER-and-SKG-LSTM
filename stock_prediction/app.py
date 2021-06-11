from stock_prediction.web_crawling import crawl_yahoo_finance
from stock_prediction.util import config
from stock_prediction.model import KnowledgeGraphEmbedding, SKGLSTMModel, OntologyNER
from stock_prediction.datasets import Datasets

import argparse


def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', '-j', type=str, dest="job", required=True, help='job to do')

    return parser.parse_args()


def run_ontology_ner():
    o_ner = OntologyNER()
    o_ner.preprocess()
    o_ner.evaluate()


def run_skg_lstm():
    for s in config.stock_list:
        df_node_train = KnowledgeGraphEmbedding(s).get_embedding_matrix()
        skg_lstm = SKGLSTMModel(df_node_train, s)
        skg_lstm.evaluate()


def generate_testing_set():
    for s in config.stock_list:
        Datasets().generate_testing_set(s)


def start_job(job):
    job_dict = {
        'crawl_yahoo_finance': crawl_yahoo_finance,
        'generate_testing_set': generate_testing_set,
        'run_ontology_ner': run_ontology_ner,
        'run_skg_lstm': run_skg_lstm
    }

    if job in [*job_dict]:
        job_dict[job]()


if __name__ == "__main__":
    args = process_command()
    start_job(args.job)
