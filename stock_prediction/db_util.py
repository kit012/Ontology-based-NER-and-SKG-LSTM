from urllib import parse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from stock_prediction.config import Config

config = Config()


def load_session():
    """Load and return a new SQLalchemy session and engine.

    Returns
    -------
    session : sqlalchemy.orm.session.Session
        Session instance.
    engine : sqlalchemy.engine.Engine
        Engine instance.
    """

    sqlalchemy_db_url = 'mysql+pymysql://' + config.mysql_username + ':' + parse.unquote_plus(
        str(config.mysql_password)) + '@' + str(config.mysql_host) + '/' + config.mysql_db_name + ''

    engine = create_engine(sqlalchemy_db_url, echo=True)
    Session = sessionmaker(bind=engine, autoflush=False)
    session = Session()
    declarative_base().metadata.create_all(engine)
    return session, engine
