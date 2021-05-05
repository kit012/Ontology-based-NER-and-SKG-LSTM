from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Sequence, insert, func, ForeignKey, and_, update, select

from stock_prediction.db_util import load_session
from sqlalchemy.dialects import mysql

SQLAlchemyBase = declarative_base()


class NewsEntity(SQLAlchemyBase):
    __tablename__ = 'TBL_NEWS_ENTITY'

    ID = Column(mysql.INTEGER(11), Sequence('user_id_seq'), primary_key=True, nullable=False)
    ENTITY_ID = Column(mysql.INTEGER(11), ForeignKey('TBL_ENTITY.ID'), nullable=False)
    NEWS_ID = Column(mysql.INTEGER(11), ForeignKey('TBL_WC_YAHOO_FINANCE.ID'), nullable=False)
    START_IDX = Column(mysql.INTEGER(11), nullable=False)
    END_IDX = Column(mysql.INTEGER(11), nullable=False)
    ENTITY_VALUE = Column(String(100), nullable=False)
    DEL_IND = Column(String(1), default="N")

    def __init__(self):
        self.sess, self.eng = load_session()

    def check_record_mark_del(self, entity_id, news_id, str_idx, end_idx):
        conn = self.eng.connect()
        stmt = select([func.count()]).where(
            and_(NewsEntity.ENTITY_ID == entity_id, NewsEntity.NEWS_ID == news_id, NewsEntity.START_IDX == str_idx,
                 NewsEntity.END_IDX == end_idx)).select_from(NewsEntity)
        return conn.execute(stmt).fetchone()

    def get_del_ind(self, entity_id, news_id, str_idx, end_idx):
        conn = self.eng.connect()
        stmt = select([NewsEntity.DEL_IND]).where(
            and_(NewsEntity.ENTITY_ID == entity_id, NewsEntity.NEWS_ID == news_id, NewsEntity.START_IDX == str_idx,
                 NewsEntity.END_IDX == end_idx))
        return conn.execute(stmt).fetchone()

    def select_entity_by_news_id(self, news_id):
        conn = self.eng.connect()
        stmt = select([NewsEntity]).where(and_(NewsEntity.NEWS_ID == news_id, NewsEntity.DEL_IND == 'N'))
        return conn.execute(stmt)

    def insert_new_record(self, record):
        conn = self.eng.connect()
        if self.check_record_mark_del(record["entity_id"], record["news_id"], record["start_idx"], record["end_idx"])[
            0] == 0:
            stmt = insert(NewsEntity).values(ENTITY_ID=record["entity_id"], NEWS_ID=record["news_id"],
                                             START_IDX=record["start_idx"], END_IDX=record["end_idx"],
                                             ENTITY_VALUE=record['entity_value'])
        elif self.get_del_ind(record["entity_id"], record["news_id"], record["start_idx"], record["end_idx"])[0] == 'N':
            return
        else:
            stmt = update(NewsEntity).where(
                and_(NewsEntity.ENTITY_ID == record["entity_id"], NewsEntity.NEWS_ID == record["news_id"],
                     NewsEntity.START_IDX == record["start_idx"], NewsEntity.END_IDX == record["end_idx"])).values(
                DEL_IND="N")
        conn.execute(stmt)
        self.sess.commit()

    def mark_del_by_entity_id_and_idx(self, news_id, entity_id, str_idx, end_idx):
        conn = self.eng.connect()
        stmt = update(NewsEntity).where(
            and_(NewsEntity.NEWS_ID == news_id, NewsEntity.ENTITY_ID == entity_id, NewsEntity.START_IDX == str_idx,
                 NewsEntity.END_IDX == end_idx)).values(DEL_IND="Y")
        conn.execute(stmt)
        self.sess.commit()
