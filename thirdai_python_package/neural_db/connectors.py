import os
import tempfile
from typing import List, Type

import pandas as pd
from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext
from sqlalchemy import ColumnCollection, MetaData, create_engine, select, text
from sqlalchemy.sql.base import ReadOnlyColumnCollection

from .utils import SUPPORTED_EXT, Credentials


class Connector:
    def __init__(self, user_creds: Credentials):
        self._user_creds = user_creds

    def connect(self):
        raise NotImplementedError()

    def next_batch(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_session(self):
        raise NotImplementedError()

    @property
    def username(self):
        return self._user_creds._username


class SQLConnector(Connector):
    BATCH_SIZE = 100_000

    def __init__(
        self,
        user_creds: Credentials,
        id_col: str,
        strong_columns: List[str],
        weak_columns: List[str],
    ):
        super().__init__(user_creds)
        self.connect(user_creds)
        self.df_iter = pd.read_sql(
            sql=self._user_creds._table_name,
            con=self._connection,
            columns=[id_col] + strong_columns + weak_columns,
            chunksize=self.BATCH_SIZE,
        )

    def get_db_url(self):
        db_url = f"postgresql://{self._user_creds._username}:{self._user_creds._password}@{self._user_creds._host}:{self._user_creds._port}/{self._user_creds._database_name}"
        return db_url
    
    def connect(self):
        db_url = self._user_creds.get_db_url()
        engine = create_engine(db_url)
        self._connection = engine.connect()

    def next_batch(self):
        for batch in self.df_iter:
            yield batch

    def get_session(self):
        return self._connection


class SharePointConnector(Connector):
    FILE_LIMIT: int = 10

    def __init__(self, user_creds: Credentials):
        super().__init__(user_creds)
        self.connect()
        self.index_table = pd.DataFrame(
            columns=["File_ID", "FileName", "Ext", "server_relative_url"]
        )
        self.index_table.set_index(keys="File_ID", inplace=True)

    def connect(self):
        creds = UserCredential(user_name=self._user_creds._username, password=self._user_creds._password)
        self._ctx = ClientContext(base_url=self._user_creds._site_url).with_credentials(
            credentials=creds
        )

        # executing dummy query to check authentication
        self._ctx.execute_query()

    def next_batch(self) -> str:
        try:
            # Sharepoint Library by it's path
            library = self._ctx.web.get_folder_by_server_relative_path(
                self._user_creds._library_path
            )
            self._ctx.load(library)

            # Get file properties from the library
            files = library.files
            self._ctx.load(files)

            # query Execution retrieving file information
            self._ctx.execute_query()

            for start_file_id in slice(0, len(files), self.FILE_LIMIT):
                batch_folder = tempfile.TemporaryDirectory()
                batched_files = files[start_file_id : start_file_id + self.FILE_LIMIT]
                for file in batched_files:
                    filename = file.properties["Name"]
                    file_server_relative_url = file.properties["ServerRelativeUrl"]

                    file_ext = filename.split(sep=".")[-1]

                    if file_ext in SUPPORTED_EXT:
                        with open(os.path.join(batch_folder.name, filename)) as fp:
                            file.download(fp).execute_query()
                    yield batch_folder
        except Exception as e:
            self.index_table.drop(labels=self.index_table.index, inplace=True)
            print(str(e))

    def get_session(self):
        return self._ctx
