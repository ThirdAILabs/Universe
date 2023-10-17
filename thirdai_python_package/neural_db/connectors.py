from typing import List, Type

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine.base import Connection as sqlConn

from .utils import SUPPORTED_EXT, Credentials


class Connector:
    def connect(self):
        raise NotImplementedError()

    def next_batch(self) -> pd.DataFrame:
        raise NotImplementedError()


class SQLConnector(Connector):
    def __init__(
        self,
        engine: sqlConn,
        columns: List[str],
        chunk_size: int,
        table_name: str,
    ):
        super().__init__()
        self._engine = engine
        self.connect()
        self.table_name = table_name
        self.columns = list(set(columns))
        self.chunk_size = chunk_size

    def execute(self, query: str, param={}):
        return self._connection.execute(statement=text(query), parameters=param)

    def get_engine_url(self):
        return self._engine.url

    def connect(self):
        self._connection = self._engine.connect()

    def next_chunk(self):
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
