from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Sequence, select

from stock_prediction.db_util import load_session
from sqlalchemy.dialects import mysql

SQLAlchemyBase = declarative_base()


class Entity(SQLAlchemyBase):
    __tablename__ = 'TBL_ENTITY'

    ID = Column(mysql.INTEGER(11), Sequence('user_id_seq'), primary_key=True, nullable=False)
    NAME = Column(String(20), nullable=True)
    ALIAS = Column(String(50), nullable=True)
    HIGHLIGHT_COLOR = Column(String(50), nullable=True)

    def __init__(self):
        self.sess, self.eng = load_session()

    def select_entity_color(self):
        conn = self.eng.connect()
        stmt = select([Entity])
        return conn.execute(stmt)