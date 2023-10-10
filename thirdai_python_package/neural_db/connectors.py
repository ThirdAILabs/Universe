from typing import List

import pandas as pd
import os
from sqlalchemy import ColumnCollection, MetaData, create_engine, select, text
from sqlalchemy.sql.base import ReadOnlyColumnCollection
from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext
from .utils import ClientCredentials, SUPPORTED_EXT
import tempfile

BATCH_SIZE = 100_000

class DataConnector:
    def __init__(self, client_credentials: ClientCredentials):
        self._client_credentials = client_credentials

    def connect(self) -> bool:
        raise NotImplementedError()

    def next_batch(self) -> pd.DataFrame:
        raise NotImplementedError()
    
    def get_session(self):
        raise NotImplementedError()

    @property
    def username(self):
        return self._client_credentials._username

    @property
    def password(self, new_password):
        self._client_credentials._password = new_password


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
    FILE_LIMIT: int = 10
    def __init__(self, client_credentials: ClientCredentials, site_url: str):
        super().__init__(client_credentials)
        self._site_url = site_url
        self.connect()
        self.index_table = pd.DataFrame(columns = ["File_ID", "FileName", "Ext", "server_relative_url"])
        self.index_table.set_index(keys = "File_ID", inplace = True)

    def connect(self) -> bool:
        creds = UserCredential(user_name = self._username, password = self._password)

        try:
            self._ctx = ClientContext(base_url=self._site_url).with_credentials(credentials = creds)

            # executing dummy query to check authentication
            self._ctx.execute_query()
            
        except Exception as e:
            print(str(e))
            return False

        return True
        
    def next_batch(self) -> str:
        try:
            # Sharepoint Library by it's path
            library = self._ctx.web.get_folder_by_server_relative_path(self._client_credentials._library_path)
            self._ctx.load(library)

            # Get file properties from the library
            files = library.files
            self._ctx.load(files)

            # query Execution retrieving file information
            self._ctx.execute_query()

            for start_file_id in slice(0, len(files), self.FILE_LIMIT):
                batch_folder = tempfile.TemporaryDirectory()
                batched_files = files[start_file_id: start_file_id + self.FILE_LIMIT]
                for file in batched_files:
                    filename = file.properties["Name"]
                    file_server_relative_url = file.properties["ServerRelativeUrl"]

                    file_ext = filename.split(sep = '.')[-1]

                    if file_ext in SUPPORTED_EXT:
                        with open(os.path.join(batch_folder.name, filename)) as fp:
                            file.download(fp).execute_query()
                    yield batch_folder
        except Exception as e:
            self.index_table.drop(labels = self.index_table.index, inplace = True)
            print(str(e))

    def get_session(self):
        return self._ctx