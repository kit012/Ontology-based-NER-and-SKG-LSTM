import os.path
import yaml
from datetime import datetime, timedelta, date
import pandas_market_calendars

ADJ_END_DATE = 10  # To ensure it can find next trading date


class Config(object):
    def __init__(self, spec_path=''):
        config_dict = read_config(spec_path)

        # Environment
        self.chrome_driver_path = config_dict['env']['chrome_driver_path']
        self.java_home = config_dict['env']['java_home']
        self.spark_home = config_dict['env']['spark_home']
        self.synonyms_path = config_dict['env']['synonyms_path']
        self.phrases_path = config_dict['env']['phrases_path']
        self.yahoo_finance_path = config_dict['env']['yahoo_finance_path']
        self.kaggle_benzinga_path = config_dict['env']['kaggle_benzinga_path']

        # Parameter
        self.threads = config_dict['para']['threads']
        self.stock_list = config_dict['para']['stock_list']
        self.start_date = config_dict['para']['start_date']
        self.end_date = config_dict['para']['end_date']
        self.cut_off_date = config_dict['para']['cut_off_date']
        self.window_size = config_dict['para']['window_size']

        # Word2vec
        self.train_word2vec = True if config_dict['word2vec']['train'] == 'True' else False
        self.word2vec_max_group_num = config_dict['word2vec']['max_group_num']
        self.word2vec_similarity_method = config_dict['word2vec']['similarity_method']
        self.word2vec_eps = config_dict['word2vec']['eps']
        self.word2vec_min_samples = config_dict['word2vec']['min_samples']

        # KGE
        self.kge_emb_dim = config_dict['kge']['emb_dim']
        self.kge_lr = config_dict['kge']['lr']
        self.kge_n_epochs = config_dict['kge']['n_epochs']
        self.kge_b_size = config_dict['kge']['b_size']
        self.kge_margin = config_dict['kge']['margin']

        # MySQL
        self.mysql_host = config_dict['mysql']['host']
        self.mysql_port = config_dict['mysql']['port']
        self.mysql_username = config_dict['mysql']['username']
        self.mysql_password = config_dict['mysql']['password']
        self.mysql_db_name = config_dict['mysql']['db_name']

        nasdaq = pandas_market_calendars.get_calendar('NASDAQ')
        trading_date_idx = nasdaq.valid_days(start_date=self.start_date,
                                             end_date=adj_days_to_str_date(self.end_date, ADJ_END_DATE))
        self.trading_date_list = [i.strftime('%Y-%m-%d') for i in trading_date_idx]


def read_config(local_path):
    default_path = 'config.yaml'

    if os.path.isfile(local_path):
        config_file_path = open(local_path)
    elif os.path.isfile(default_path):
        config_file_path = open(default_path)
    else:
        print('config file not found!')
        raise FileNotFoundError
    if config_file_path is not None:
        return yaml.load(config_file_path, Loader=yaml.FullLoader)
    else:
        raise SystemExit


def adj_days_to_str_date(the_date, days):
    if isinstance(the_date, date):
        return datetime.strftime(the_date + timedelta(days=days), '%Y-%m-%d')
    elif isinstance(the_date, str):
        return datetime.strftime(datetime.strptime(the_date, '%Y-%m-%d').date() + timedelta(days=days), '%Y-%m-%d')
    else:
        print('The type of date should either date type or string type.')
        raise Exception
