import os
import tempfile
from typing import List, Type

import pandas as pd
# from office365.sharepoint.client_context import ClientContext
from sqlalchemy import inspect, text
from sqlalchemy.engine.base import Connection as sqlConn
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.base import ReadOnlyColumnCollection

from .utils import SUPPORTED_EXT, Credentials


class Connector:
    def connect(self):
        raise NotImplementedError()

    def next_batch(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_session(self):
        raise NotImplementedError()

    def execute(self, query: str):
        return self._connection.execute(text(query))


class SQLConnector(Connector):
    def __init__(
        self,
        engine: sqlConn,
        id_col: str,
        strong_columns: List[str],
        weak_columns: List[str],
        reference_columns: List[str],
        batch_size: int,
        table_name: str,
    ):
        super().__init__()
        self._engine = engine
        self.connect()
        columns = list(set([id_col] + strong_columns + weak_columns + reference_columns))
        self.df_iter = pd.read_sql(
            sql=table_name,
            con=self._connection,
            columns=columns,
            chunksize=batch_size,
        )

    def get_engine_url(self):
        return self._engine.url

    def connect(self):
        self._connection = self._engine.connect()

    def next_batch(self):
        try:
            return next(self.df_iter)
        except StopIteration:
            return None

    def get_session(self):
        Session = sessionmaker(bind=self._engine)
        yield Session()


# class SharePointConnector(Connector):
#     FILE_LIMIT: int = 10

#     def __init__(self, user_creds: Credentials):
#         super().__init__(user_creds)
#         self.connect()
#         self.index_table = pd.DataFrame(
#             columns=["File_ID", "FileName", "Ext", "server_relative_url"]
#         )
#         self.index_table.set_index(keys="File_ID", inplace=True)

#     def connect(self):
#         creds = UserCredential(
#             user_name=self._user_creds._username, password=self._user_creds._password
#         )
#         self._ctx = ClientContext(base_url=self._user_creds._site_url).with_credentials(
#             credentials=creds
#         )

#         # executing dummy query to check authentication
#         self._ctx.execute_query()

#     def next_batch(self) -> str:
#         try:
#             # Sharepoint Library by it's path
#             library = self._ctx.web.get_folder_by_server_relative_path(
#                 self._user_creds._library_path
#             )
#             self._ctx.load(library)

#             # Get file properties from the library
#             files = library.files
#             self._ctx.load(files)

#             # query Execution retrieving file information
#             self._ctx.execute_query()

#             for start_file_id in slice(0, len(files), self.FILE_LIMIT):
#                 batch_folder = tempfile.TemporaryDirectory()
#                 batched_files = files[start_file_id : start_file_id + self.FILE_LIMIT]
#                 for file in batched_files:
#                     filename = file.properties["Name"]
#                     file_server_relative_url = file.properties["ServerRelativeUrl"]

#                     file_ext = filename.split(sep=".")[-1]

#                     if file_ext in SUPPORTED_EXT:
#                         with open(os.path.join(batch_folder.name, filename)) as fp:
#                             file.download(fp).execute_query()
#                     yield batch_folder
#         except Exception as e:
#             self.index_table.drop(labels=self.index_table.index, inplace=True)
#             print(str(e))

#     def get_session(self):
#         return self._ctx
