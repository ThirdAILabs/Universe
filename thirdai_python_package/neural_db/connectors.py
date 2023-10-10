from typing import List

import pandas as pd
from sqlalchemy import ColumnCollection, MetaData, create_engine, select, text
from sqlalchemy.sql.base import ReadOnlyColumnCollection
from .utils import ClientCredentials

BATCH_SIZE = 100_000

class DataConnector:
    def __init__(self, client_credentials: ClientCredentials):
        self._username = client_credentials.username
        self._password = client_credentials.password
        self._uri = client_credentials.uri
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
    def __init__(self, client_credentials: ClientCredentials, table_name: str, id_col: str, strong_columns: List[str], weak_columns: List[str]):
        self.connect(client_credentials)
        self.df_iter = pd.read_sql(sql = table_name, con=self.connection, columns=[id_col] + strong_columns + weak_columns, chunksize=BATCH_SIZE)
            
    def connect(self, client_credentials: ClientCredentials):
        db_url = client_credentials.get_db_url
        engine = create_engine(db_url)
        self.connection = engine.connect()
        
    def next_batch(self):
        for batch in self.df_iter:
            yield batch
    
    def get_session(self):
        return self.connection
        
            
class SharePointConnector(DataConnector):
    pass
