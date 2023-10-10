from typing import List

import pandas as pd
from sqlalchemy import ColumnCollection, MetaData, create_engine, select, text
from sqlalchemy.sql.base import ReadOnlyColumnCollection

BATCH_SIZE = 100_000


class Credentials:
    def __init__(
        self, username: str, password: str, host: str, port: int, database_name: str
    ):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database_name = database_name

    def get_db_url(self):
        db_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"
        return db_url


class DataConnector:
    def __init__(self, username: str, password: str, uri: str):
        self._username = username
        self._password = password
        self._uri = uri
        self._session = None

    def connect() -> bool:
        raise NotImplementedError()

    def next_batch(self):
        raise NotImplementedError()

    def get_session(self):
        return self._session

    @property
    def username(self):
        return self._username

    @property
    def password(self, new_password):
        self._password = new_password


class SQLConnector(DataConnector):
    def __init__(self, auth_options: Credentials, table_name: str, id_col: str, strong_columns: List[str], weak_columns: List[str]):
        self.connect(auth_options)
        self.df_iter = pd.read_sql(sql = table_name, con=self.connection, columns=[id_col] + strong_columns + weak_columns, chunksize=BATCH_SIZE)
            
    def connect(self, auth_options: Credentials):
        db_url = auth_options.get_db_url
        engine = create_engine(db_url)
        self.connection = engine.connect()
        
    def next_batch(self):
        for batch in self.df_iter:
            yield batch
    
    def get_session(self):
        return self.connection
        
            
class SharePointConnector(DataConnector):
    pass
