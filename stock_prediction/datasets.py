import yfinance as yf
from pypfopt.expected_returns import returns_from_prices
import pandas as pd
import swifter

from stock_prediction.config import adj_days_to_str_date
from stock_prediction.util import config, use_post_date_if_available, find_nearest_trading_date


def convert_price2movement(df, col):
    df[col + '_FLAG'] = df[col].apply(lambda x: 0 if x < 0 else 1)
    df.drop(columns=col, inplace=True)


def get_stock_price_data():
    return yf.download(config.stock_list,
                       adj_days_to_str_date(config.start_date, -1),
                       adj_days_to_str_date(config.end_date, 1))


def get_movement():
    stock_return = returns_from_prices(get_stock_price_data()['Adj Close'], log_returns=True)

    movement = stock_return.reset_index().rename(columns={0: 'FLAG'})
    movement['Date'] = movement['Date'].astype(str)

    for stock in config.stock_list:
        convert_price2movement(movement, stock)

    return movement


class Datasets(object):
    def __init__(self):
        self.stock_list = config.stock_list
        self.window_size = config.window_size
        self.yahoo_finance_path = config.yahoo_finance_path
        self.kaggle_benzinga_path = config.kaggle_benzinga_path
        self.cut_off_date = config.cut_off_date
        self.end_date = config.end_date

    def get_yahoo_finance(self, drop_duplicates_only=False):
        df = pd.read_csv(self.yahoo_finance_path)
        df['TITLE'] = df['TITLE'].apply(lambda x: x.replace('\xa0', ''))

        if drop_duplicates_only:
            df = df.drop(columns=['ID', 'UP_DT'])
        else:
            df['DT'] = df.swifter.apply(lambda x: use_post_date_if_available(x['UP_DT'], x['POST_DT']), axis=1)
            df.drop(columns=['ID', 'CATEGORY', 'LINK', 'AUTHOR', 'SUMMARY', 'CONTENT', 'UP_DT', 'POST_DT'],
                    inplace=True)
            df.set_index('DT', inplace=True)
            df.sort_index(inplace=True)
        return df.drop_duplicates()

    def get_training_set(self):
        return self.get_yahoo_finance(drop_duplicates_only=False).loc[:str(self.cut_off_date)]

    def get_kaggle_benzinga_data_set(self):
        return pd.read_csv(self.kaggle_benzinga_path)

    def generate_testing_set(self, stock_code):
        df = self.get_yahoo_finance()

        df = df.loc[self.cut_off_date:self.end_date]
        df = df.reset_index()
        df['PRE_DT'] = df['DT'].swifter.apply(find_nearest_trading_date)
        df['DT'] = df['DT'].apply(lambda x: x.date())
        df = df[~df.TAG.isna()]
        df['TAG'] = df['TAG'].apply(lambda x: x.split(','))
        df = df.explode('TAG')
        df = df[df.TAG == stock_code][['DT', 'TITLE', 'PRE_DT']]
        df = df.groupby(['DT', 'PRE_DT'])['TITLE'].apply(list).reset_index()

        stock_price_data = yf.download(stock_code, adj_days_to_str_date(config.trading_date_list[0], -1),
                                       config.trading_date_list[-1])
        stock_return = returns_from_prices(stock_price_data['Adj Close'])
        stock_return = stock_return.reset_index()
        stock_return['Date'] = stock_return['Date'].astype(str)

        df = pd.merge(df, stock_return, left_on='PRE_DT', right_on='Date', how='left').drop(columns=['Date']).rename(
            columns={'Adj Close': stock_code})

        df[stock_code + '_FLAG'] = df[stock_code].apply(lambda x: 0 if x < 0 else 1)

        print('Generating ' + stock_code)
        df.to_parquet(
            'out/testing_set/testing_news_' + str(self.window_size) + 'd_' + stock_code + '.parquet')
