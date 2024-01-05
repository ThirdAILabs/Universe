# Python
from abc import ABC, abstractmethod
from typing import List, Generator, Tuple
import uuid
from pathlib import Path
import shutil

# Libraries
import pandas as pd
import sqlite3

# Local
from .constraint_matcher import Filter


class Table(ABC):
    @property
    @abstractmethod
    def columns(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    @abstractmethod
    def ids(self) -> List[int]:
        pass

    @abstractmethod
    def field(self, row_id: int, column: str):
        pass

    @abstractmethod
    def row_as_dict(self, row_id: int) -> dict:
        pass

    @abstractmethod
    def range_rows_as_dicts(self, from_row_id: int, to_row_id: int) -> List[dict]:
        pass

    @abstractmethod
    def iter_rows_as_dicts(self) -> Generator[Tuple[int, dict], None, None]:
        pass

    @abstractmethod
    def apply_filter(self, filterer: Filter, column_name: str):
        pass

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass


class DataFrameTable(Table):
    def __init__(self, df: pd.DataFrame, id_column: str):
        self.df = df
        # TODO: Reset index first?
        self.df = self.df.set_index(id_column)

    @property
    def columns(self) -> List[str]:
        return self.df.columns

    @property
    def size(self) -> int:
        return len(self.df)

    @property
    def ids(self) -> List[int]:
        return self.df.index.to_list()

    def field(self, row_id: int, column: str):
        return self.df[column].loc[row_id]

    def row_as_dict(self, row_id: int) -> dict:
        return self.df.loc[row_id].to_dict()

    def range_rows_as_dicts(self, from_row_id: int, to_row_id: int) -> List[dict]:
        return self.df.loc[from_row_id:to_row_id].to_dict(orient="records")

    def iter_rows_as_dicts(self) -> Generator[Tuple[int, dict], None, None]:
        for row_id, row in self.df.iterrows():
            yield (row_id, row.to_dict())

    def apply_filter(self, filterer: Filter):
        filterer.filter_df_column(self.df)


class SQLiteTable(Table):
    def __init__(self, df: pd.DataFrame, id_column: str):
        # TODO: Reset index first?
        self.db_path = f"{uuid.uuid4()}.db"
        self.id_column = id_column
        self.db_size = len(df)
        self.db_columns = [col for col in df.columns if col != id_column]
        # We don't save the db connection and instead create a new connection
        # each time to simplify serialization.
        con = sqlite3.connect(self.db_path)
        df.to_sql(name="sqlitetable", con=con)

    @property
    def columns(self) -> List[str]:
        return self.db_columns

    @property
    def size(self) -> int:
        return self.db_size

    @property
    def ids(self) -> List[int]:
        con = sqlite3.connect(self.db_path)
        return pd.read_sql(f"select {self.id_column} from sqlitetable", con)[
            self.id_column
        ]

    def field(self, row_id: int, column: str):
        con = sqlite3.connect(self.db_path)
        return pd.read_sql(
            f"select {column} from sqlitetable where {self.id_column}=={row_id}",
            con,
        )[column][0]

    def row_as_dict(self, row_id: int) -> dict:
        con = sqlite3.connect(self.db_path)
        return pd.read_sql(
            f"select * from sqlitetable where {self.id_column}=={row_id}", con
        ).to_dict("records")[0]

    def range_rows_as_dicts(self, from_row_id: int, to_row_id: int) -> List[dict]:
        con = sqlite3.connect(self.db_path)
        return pd.read_sql(
            f"select * from sqlitetable where {self.id_column}>={from_row_id} and {self.id_column}<{to_row_id}",
            con,
        ).to_dict("records")

    def iter_rows_as_dicts(self) -> Generator[Tuple[int, dict], None, None]:
        size = self.size
        chunk_size = 1000  # Hardcoded for now
        # Load in chunks
        for chunk_start in range(0, size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, size)
            for row in self.range_rows_as_dicts(chunk_start, chunk_end):
                yield row[self.id_column], row

    def save_meta(self, directory: Path):
        shutil.copy(self.db_path, directory / Path(self.db_path).name)

    def load_meta(self, directory: Path):
        self.db_path = str(directory / Path(self.db_path).name)

    def apply_filter(self, filterer: Filter):
        filterer.filter_sql_column(sqlite3.connect(self.db_path), "sqlitetable")
