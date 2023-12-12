import os
import shutil
import tempfile
from typing import List, Optional

import pandas as pd
from office365.sharepoint.client_context import ClientContext
from simple_salesforce import Salesforce
from sqlalchemy import inspect, text
from sqlalchemy.engine.base import Connection as sqlConn

from .utils import DIRECTORY_CONNECTOR_SUPPORTED_EXT


class Connector:
    def chunk_iterator(self):
        raise NotImplementedError()


class SQLConnector(Connector):
    def __init__(
        self,
        engine: sqlConn,
        table_name: str,
        id_col: str,
        columns: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ):
        self._engine = engine
        self.id_col = id_col
        self.columns = columns
        self.table_name = table_name
        self.chunk_size = chunk_size
        self._connection = self._engine.connect()

    def __del__(self):
        self._connection.close()

    def execute(self, query: str, param={}):
        result = self._connection.execute(statement=text(query), parameters=param)
        return result

    def get_engine_url(self):
        return self._engine.url

    def chunk_iterator(self):
        return pd.read_sql(
            sql=f"SELECT {', '.join(self.columns)} FROM {self.table_name} ORDER BY {self.id_col}",
            con=self._connection,
            chunksize=self.chunk_size,
        )

    def total_rows(self):
        return self.execute(query=f"select count(*) from {self.table_name}").fetchone()[
            0
        ]

    def cols_metadata(self):
        inspector = inspect(self._engine)
        return inspector.get_columns(self.table_name)

    def get_all_rows(self, cols: List[str] = "*"):
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
        chunk_size: int = 10485760,
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
            self._files = list(
                filter(
                    lambda file: file.properties["Name"].split(sep=".")[-1]
                    in DIRECTORY_CONNECTOR_SUPPORTED_EXT,
                    self._files,
                )
            )

            if not len(self._files) > 0:
                raise FileNotFoundError("No files of supported extension is present")

            # we need to maintain a fixed order of files because local_docs needs to be also equivalent as detailed in the test cases. For more info: ndb_utls.py::build_local_sharepoint_doc & test_connector_document_implementation.py
            self._files = sorted(self._files, key=lambda file: file.properties["Name"])
        except Exception as e:
            print("Unable to retrieve files from SharePoint, Error: " + str(e))

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
        except Exception as e:
            print("Unable to retrieve file(s) from SharePoint, Error: " + str(e))
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

    def num_files(self):
        return len(self._files)


class SalesforceConnector(Connector):
    def __init__(
        self,
        instance: Salesforce,
        object_name: str,
        fields: Optional[List[str]] = None,
    ) -> None:
        self._instance = instance
        self._object_name = object_name
        self._fields = fields

    def execute(self, query: str):
        # Returns an OrderedDicts with keys ['totalSize', 'done', 'records']
        return self._instance.query(query)

    def chunk_iterator(self):
        query = f"SELECT {', '.join(self._fields)} FROM {self._object_name}"
        results = self._instance.bulk.__getattr__(self._object_name).query(
            query, lazy_operation=True
        )
        for chunk in results:
            # Number of records in each chunk can atmost 10K (can't be changed with salesforce bulk API).
            chunk_df = pd.DataFrame(chunk)
            chunk_df.drop(columns=["attributes"], inplace=True)
            yield chunk_df

    def total_rows(self):
        result = self.execute(query=f"SELECT COUNT() from {self._object_name}")
        return result["totalSize"]

    def field_metadata(self):
        object_schema = self._instance.__getattr__(self._object_name).describe()
        return object_schema["fields"]

    @property
    def sf_instance(self):
        return self._instance.sf_instance

    @property
    def base_url(self):
        return self._instance.base_url
