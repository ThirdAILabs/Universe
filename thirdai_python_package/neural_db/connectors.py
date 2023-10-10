from typing import List

import pandas as pd
from sqlalchemy import ColumnCollection, MetaData, create_engine, select, text
from sqlalchemy.sql.base import ReadOnlyColumnCollection
from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext

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
    def __init__(self, username: str, password: str):
        self._username = username
        self._password = password

    def connect() -> bool:
        raise NotImplementedError()

    def next_batch(self):
        raise NotImplementedError()

    @property
    def username(self):
        return self._username

    @property
    def password(self, new_password):
        self._password = new_password


class SQLConnector(DataConnector):
    def __init__(self, auth_options: Credentials, table_name: str, id_col: str, strong_columns: List[str], weak_columns: List[str]):
        self.connect(auth_options)
        self.query_cols = id_col + strong_columns + weak_columns
        self.df_iter = pd.read_sql(sql = table_name, con=self.connection, columns=[id_col] + strong_columns + weak_columns, chunksize=BATCH_SIZE)
            
    def connect(self, auth_options: Credentials):
        db_url = auth_options.get_db_url
        engine = create_engine(db_url)
        metadata = MetaData()
        metadata.reflect(bind=engine)
        
        metadata = metadata
        self.connection = engine.connect()
        
    def next_batch(self):
        for batch in self.df_iter:
            yield batch
    
    def get_session(self):
        return self.connection
        
            
class SharePointConnector(DataConnector):
    def __init__(self, username: str, password: str, site_url: str):
        super().__init__(username, password)
        self._site_url = site_url
        self.connect()

    def connect(self) -> bool:
        creds = UserCredential(user_name = self._username, password = self._password)

        try:
            self._ctx = ClientContext(base_url=self._site_url).with_credentials(credentials = creds)

            #dummy query execute to check authentication
            self._ctx.execute_query()
            
        except Exception as e:
            print(str(e))
            return False

        return True
        

    def next_batch(self):
        raise NotImplementedError()

    def get_session(self):
        return self._session