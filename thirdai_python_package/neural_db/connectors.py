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
        self.connection = self.connect(auth_options)
        self.table_name = table_name
        self.query_cols = id_col + strong_columns + weak_columns
        self.total_rows = self.connection.execute(text(f'select count(*) from {self.table_name}')).fetchone()[0]
        table = self.metadata.tables[self.config['table_name']]
        self.offset = 0
        self.cols = ColumnCollection([(col, getattr(table.c, col)) for col in self.query_cols])  
        
    def connect(self, auth_options: Credentials):
        db_url = auth_options.get_db_url
        self.engine = create_engine(db_url)
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        
        self.metadata = metadata
        self.connection = self.engine.connect()
        
    def next_batch(self):

        if (self.offset < self.total_rows):
            select_query = select(ReadOnlyColumnCollection(self.cols)).offset(offset).limit(BATCH_SIZE)
            result = self.connection.execute(select_query)
            
            offset+=BATCH_SIZE
            for row in result:
                new_row_df = pd.DataFrame([row], columns=df.columns)
                df = pd.concat([df, new_row_df], ignore_index=True)
        
            
class SharePointConnector(DataConnector):
    pass
        
        
        
        
        
    
    