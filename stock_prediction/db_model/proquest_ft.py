from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, String, Sequence, insert, update, select, and_
from stock_prediction.db_util import load_session
from sqlalchemy.dialects import mysql

SQLAlchemyBase = declarative_base()


class ProquestFt(SQLAlchemyBase):
    __tablename__ = 'TBL_PROQUEST_FT'

    ID = Column(mysql.INTEGER(11), Sequence('user_id_seq'), primary_key=True, nullable=False)
    TITLE = Column(mysql.LONGTEXT)
    UP_DT = Column(DateTime)
    PUB_DT = Column(mysql.LONGTEXT)
    CONTENT = Column(mysql.LONGTEXT)
    DETAILS = Column(mysql.LONGTEXT)

    def __init__(self):
        self.sess, self.eng = load_session()

    def __repr__(self):
        return "<User(ID='%s', UP_DT='%s', CATEGORY='%s', SRC='%s', TITLE='%s', TAG='%s')>" % (
            self.ID, self.TITLE, self.UP_DT, self.PUB_DT, self.CONTENT, self.DETAILS)

    def insert_new_records_batch(self, records):
        conn = self.eng.connect()
        stmt = insert(ProquestFt).values(records)
        conn.execute(stmt)
        self.sess.commit()
