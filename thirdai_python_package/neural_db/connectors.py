from sqlalchemy import create_engine, ColumnCollection, select, text, MetaData
from sqlalchemy.sql.base import ReadOnlyColumnCollection
from typing import List
import pandas as pd

BATCH_SIZE = 100_000

class Credentials:
    def __init__(self, username: str, password: str, host: str, port: int, database_name: str):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database_name = database_name

    def get_db_url(self):
        db_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"
        return db_url
    
class DataConnector:
  
    def __init__(self, username, password, URI):
        pass
    
    def connect()-> bool:
        # establish the connection with the (username, password)
        pass
    
    def process_data(self);
        """
        Process the data in batch using nextbatch() function
        """
        pass
    
    def next_batch(self):
        pass

    def get_session():
        """
        Returns the connection session
        """

    @property
    def username(self):
        raise NotImplementedError()

    @property
    def password(self, new_password):
        raise NotImplementedError()


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
    pass
        
        
        
        
        
    
    