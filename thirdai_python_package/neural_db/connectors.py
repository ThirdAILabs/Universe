import os
import shutil
import tempfile
from typing import List, Optional

import pandas as pd
from office365.sharepoint.client_context import ClientContext
from sqlalchemy import inspect, text
from sqlalchemy.engine.base import Connection as sqlConn

from .utils import SUPPORTED_EXT


class Connector:
    def chunk_iterator(self):
        raise NotImplementedError()


class SQLConnector(Connector):
    def __init__(
        self,
        engine: sqlConn,
        columns: List[str],
        table_name: str,
        chunk_size: Optional[int] = None,
    ):
        self._engine = engine
        self.columns = list(set(columns))
        self.table_name = table_name
        self.chunk_size = chunk_size
        self._connection = self._engine.connect()

    def execute(self, query: str, param={}):
        result = self._connection.execute(statement=text(query), parameters=param)
        return result

    def get_engine_url(self):
        return self._engine.url

    def chunk_iterator(self):
        return pd.read_sql(
            sql=self.table_name,
            con=self._connection,
            columns=self.columns,
            chunksize=self.chunk_size,
        )

    def total_rows(self):
        return self.execute(query=f"select count(*) from {self.table_name}").fetchone()[
            0
        ]

    def cols_metadata(self):
        inspector = inspect(self._engine)
        return inspector.get_columns(self.table_name)

    def get_rows(self, cols: List[str] = "*"):
        if isinstance(cols, list):
            cols = ", ".join(cols)
        return self.execute(query=f"SELECT {cols} from {self.table_name}")

    def get_primary_keys(self):
        inspector = inspect(self._engine)
        pk_constraint = inspector.get_pk_constraint(self.table_name)
        return pk_constraint["constrained_columns"]


class SharePointConnector(Connector):
    def __init__(
        self,
        ctx: ClientContext,
        library_path: str,
        chunk_size: int = 10485760,  # Each file is being treated as a chunk
    ):
        self._ctx = ctx
        self.library_path = library_path
        self.chunk_size = chunk_size
        try:
            # Loading the Sharepoint library's metadata by it's path
            library = self._ctx.web.get_folder_by_server_relative_path(
                self.library_path
            )
            self._ctx.load(library)

            # Retreiving all the file's metadata from the library
            self._files = library.files
            self._ctx.load(self._files)
            self._ctx.execute_query()

            # filtering to only contain files of supported extensions
            exts = SUPPORTED_EXT[:]
            exts.remove("csv")
            self._files = list(
                filter(
                    lambda file: file.properties["Name"].split(sep=".")[-1]
                    in exts,
                    self._files,
                )
            )
            self.total_files = len(self._files)
            if not self.total_files > 0:
                raise FileNotFoundError("No files of supported extension is present")
        except Exception as e:
            print("Unable to retrieve files from SharePoint, Error0: " + str(e))

    def chunk_iterator(self):
        try:
            files_dict = {}
            temp_dir = tempfile.mkdtemp()
            currently_occupied = 0

            for file in self._files:
                file_size = int(file.properties["Length"])
                filename = file.properties["Name"]
                file_server_relative_url = file.properties["ServerRelativeUrl"]
                if (
                    len(files_dict) > 0
                    and file_size + currently_occupied >= self.chunk_size
                ):
                    # Return the fetched files
                    yield files_dict
                    files_dict.clear()
                    currently_occupied = 0
                    shutil.rmtree(temp_dir)

                    temp_dir = tempfile.mkdtemp()
                else:
                    filepath = os.path.join(temp_dir, filename)
                    with open(filepath, "wb") as fp:
                        file.download(fp).execute_query()
                        files_dict[file_server_relative_url] = filepath
                        currently_occupied += file_size
            if len(files_dict) > 0:
                yield files_dict
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @property
    def url(self):
        web = self._ctx.web.get().execute_query()
        return web.url

    @property
    def site_name(self):
        return self.url.split(sep="/")[-1]
