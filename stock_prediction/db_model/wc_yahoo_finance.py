from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, String, Sequence, insert, update, func, select, and_
from stock_prediction.db_util import load_session
from sqlalchemy.dialects import mysql

from datetime import datetime

SQLAlchemyBase = declarative_base()


class WCYahooFinance(SQLAlchemyBase):
    __tablename__ = 'TBL_WC_YAHOO_FINANCE'

    ID = Column(mysql.INTEGER(11), Sequence('user_id_seq'), primary_key=True, nullable=False)
    UP_DT = Column(DateTime)
    CATEGORY = Column(String(30))
    SRC = Column(String(100))
    POST_DT = Column(DateTime)
    LINK = Column(mysql.LONGTEXT)
    AUTHOR = Column(String(100))
    TITLE = Column(String(200))
    SUMMARY = Column(mysql.LONGTEXT)
    CONTENT = Column(mysql.LONGTEXT)
    TAG = Column(String(100))

    def __init__(self):
        self.sess, self.eng = load_session()

    def __repr__(self):
        return "<User(ID='%s', UP_DT='%s', CATEGORY='%s', SRC='%s', TITLE='%s', TAG='%s')>" % (
            self.ID, self.UP_DT, self.CATEGORY, self.SRC, self.TITLE, self.TAG)

    def check_record_exist(self, link):
        conn = self.eng.connect()
        stmt = select([func.count()]) \
            .where(WCYahooFinance.LINK == link).select_from(WCYahooFinance)
        return conn.execute(stmt)

    def check_tag_exists(self, tag, link):
        conn = self.eng.connect()
        stmt = select([WCYahooFinance.TAG]).where(WCYahooFinance.LINK == link)
        res = conn.execute(stmt)
        exist_tag = res.first()
        if exist_tag is not None:
            if exist_tag[0] is not None:
                if tag not in exist_tag[0].split(','):
                    return exist_tag[0] + ',' + tag
                else:
                    return 'skip'
        else:
            return tag

    def get_next_id(self, id):
        conn = self.eng.connect()
        stmt = select([func.max(WCYahooFinance.ID)]).select_from(WCYahooFinance)
        max_id = conn.execute(stmt).fetchone()[0]

        if id != max_id:
            while True:
                id += 1
                conn = self.eng.connect()
                stmt = select([func.count()]).where(WCYahooFinance.ID == id).select_from(WCYahooFinance)
                conn.execute(stmt)
                self.sess.commit()

                if conn.execute(stmt).fetchone()[0] != 0:
                    return id
        else:
            return max_id

    def insert_new_records_batch(self, records):
        conn = self.eng.connect()
        stmt = insert(WCYahooFinance).values(records)
        conn.execute(stmt)
        self.sess.commit()

    def select_news_by_id(self, id):
        conn = self.eng.connect()
        stmt = select([WCYahooFinance.ID, WCYahooFinance.TITLE, WCYahooFinance.POST_DT, WCYahooFinance.UP_DT,
                       WCYahooFinance.LINK, WCYahooFinance.CONTENT, WCYahooFinance.SUMMARY, WCYahooFinance.TAG,
                       WCYahooFinance.SRC]).where(WCYahooFinance.ID == id)
        conn.execute(stmt)
        self.sess.commit()
        return conn.execute(stmt)

    def select_news_to_crawl(self, num, source=None):
        conn = self.eng.connect()
        if source is not None:
            stmt = select([WCYahooFinance.ID, WCYahooFinance.SRC, WCYahooFinance.LINK]) \
                .where(and_(WCYahooFinance.CONTENT == None,
                            WCYahooFinance.SRC == source)).limit(num)
        else:
            stmt = select([WCYahooFinance.ID, WCYahooFinance.SRC, WCYahooFinance.LINK]) \
                .where(and_(WCYahooFinance.CONTENT == None,
                            WCYahooFinance.LINK.startswith('/news/'))).limit(num)
        return conn.execute(stmt)

    def select_news_wo_post_date(self, num=100):
        conn = self.eng.connect()
        # stmt = select([WCYahooFinance.LINK]) \
        #         .where(and_(WCYahooFinance.POST_DT == None, WCYahooFinance.SRC != 'Bloomberg')).order_by(WCYahooFinance.UP_DT.desc()).limit(num)
        stmt = select([WCYahooFinance.LINK]) \
            .where(WCYahooFinance.POST_DT == None).order_by(WCYahooFinance.UP_DT.desc()).limit(num)
        return conn.execute(stmt)

    def select_distinct_category(self):
        conn = self.eng.connect()
        stmt = select([WCYahooFinance.CATEGORY]).distinct()
        return conn.execute(stmt)

    def select_distinct_tag(self):
        conn = self.eng.connect()
        stmt = select([WCYahooFinance.TAG]).distinct()
        return conn.execute(stmt)

    def select_distinct_author(self):
        conn = self.eng.connect()
        stmt = select([WCYahooFinance.AUTHOR]).distinct()
        return conn.execute(stmt)

    def select_title_p_dt_by_criterion(self, page, page_size, src=None, author=None, tag=None, cat=None,
                                       post_str_dt=None, post_end_dt=None):

        criterion = None
        result_count = 0
        if src is not None:
            for s in src:
                if criterion is None:
                    criterion = WCYahooFinance.SRC == s
                else:
                    criterion = and_(criterion, WCYahooFinance.SRC == s)
        if author is not None:
            for a in author:
                if criterion is None:
                    criterion = WCYahooFinance.AUTHOR == a
                else:
                    criterion = and_(criterion, WCYahooFinance.AUTHOR == a)
        if tag is not None:
            for t in tag:
                if criterion is None:
                    criterion = WCYahooFinance.TAG == t
                else:
                    criterion = and_(criterion, WCYahooFinance.TAG == t)
        if cat is not None:
            for c in cat:
                if criterion is None:
                    criterion = WCYahooFinance.CATEGORY == c
                else:
                    criterion = and_(criterion, WCYahooFinance.CATEGORY == c)
        if post_str_dt is not None:
            if criterion is None:
                criterion = WCYahooFinance.POST_DT >= datetime.strptime(post_str_dt, '%Y-%m-%d')
            else:
                criterion = and_(criterion, WCYahooFinance.POST_DT >= datetime.strptime(post_str_dt, '%Y-%m-%d'))
        if post_end_dt is not None:
            if criterion is None:
                criterion = WCYahooFinance.POST_DT <= datetime.strptime(post_end_dt, '%Y-%m-%d')
            else:
                criterion = and_(criterion, WCYahooFinance.POST_DT <= datetime.strptime(post_end_dt, '%Y-%m-%d'))

        conn = self.eng.connect()
        if criterion is not None:
            stmt = select(
                [WCYahooFinance.ID, WCYahooFinance.TITLE, WCYahooFinance.POST_DT, WCYahooFinance.UP_DT]).where(
                criterion).order_by(WCYahooFinance.ID).offset(page * page_size).limit(page_size)

        else:
            stmt = select(
                [WCYahooFinance.ID, WCYahooFinance.TITLE, WCYahooFinance.POST_DT, WCYahooFinance.UP_DT]).order_by(
                WCYahooFinance.ID).offset(page * page_size).limit(page_size)

        conn2 = self.eng.connect()
        if criterion is not None:
            stmt2 = select([func.count()]).where(
                criterion).select_from(WCYahooFinance)
        else:
            stmt2 = select([func.count()]).select_from(WCYahooFinance)

        return conn.execute(stmt), conn2.execute(stmt2)

    def update_tag_by_link(self, tag, link):
        conn = self.eng.connect()
        tag = self.check_tag_exists(tag, link)
        if tag != 'skip':
            stmt = update(WCYahooFinance).where(WCYahooFinance.LINK == link).values(TAG=tag)
            conn.execute(stmt)
            self.sess.commit()

    def update_post_date_by_link(self, post_date, link):
        conn = self.eng.connect()
        stmt = update(WCYahooFinance).where(WCYahooFinance.LINK == link).values(POST_DT=post_date)
        conn.execute(stmt)
        self.sess.commit()

    def update_crawl_fail_by_id(self, id):
        conn = self.eng.connect()
        stmt = update(WCYahooFinance).where(WCYahooFinance.ID == id).values(CONTENT='fail to crawl')
        conn.execute(stmt)
        self.sess.commit()

    def update_news_article_by_id(self, id, post_time, author, article_text):
        conn = self.eng.connect()
        stmt = update(WCYahooFinance).where(WCYahooFinance.ID == id).values(POST_DT=post_time, AUTHOR=author,
                                                                            CONTENT=article_text)
        conn.execute(stmt)
        self.sess.commit()

    def update_stock_code_by_news_id(self, id, stock_code):
        conn = self.eng.connect()
        stmt = select([WCYahooFinance.TAG]).where(WCYahooFinance.ID == id)
        current_stock_code = conn.execute(stmt).fetchone()[0]

        # print(current_stock_code)
        # print(stock_code)

        if current_stock_code != stock_code:
            if ',' in stock_code:
                sc = stock_code.split(',')
                sc.sort()
                sc = ','.join(sc)
            else:
                sc = stock_code
            conn2 = self.eng.connect()
            stmt2 = update(WCYahooFinance).where(WCYahooFinance.ID == id).values(TAG=sc)
            conn2.execute(stmt2)
            self.sess.commit()