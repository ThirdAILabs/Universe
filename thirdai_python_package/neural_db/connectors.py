from sqlalchemy import create_engine, ColumnCollection, select, text, MetaData
from sqlalchemy.sql.base import ReadOnlyColumnCollection
from typing import List

BATCH_SIZE = 100_000

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
    def __init__(self):
        self.connection = None
        
    def connect(self, username: str, password: str, host: str, port: int, database_name: str):
        db_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
        self.engine = create_engine(db_url)
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        
        self.metadata = metadata
        self.connection = self.engine.connect()
        
    def next_batch(self):

        while(self.offset < self.total_rows):
            select_query = select(ReadOnlyColumnCollection(self.cols)).offset(offset).limit(BATCH_SIZE)
            result = self.connection.execute(select_query)
            
            offset+=BATCH_SIZE
            
            yield result
            
    def process_data(self, table_name: str, id_col: str, strong_columns: List[str], weak_columns: List[str]):
        table = self.metadata.tables[table_name]
        self.total_rows = self.connection.execute(text(f'select count(*) from {table_name}')).fetchone()[0]

        self.offset = 0
        self.cols = ColumnCollection([(id_col, getattr(table.c, id_col))] + [(col, getattr(table.c, col)) for col in strong_columns] + [(col, getattr(table.c, col)) for col in weak_columns])  
        
        for batch in self.next_batch():

            for row in batch:
                print(row[:1])
        
            
class SharePointConnector(DataConnector):
    pass
        
        
        
        
        
    
    