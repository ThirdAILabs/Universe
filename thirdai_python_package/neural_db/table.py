# Python
import shutil
import sqlite3
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Tuple

# Libraries
import pandas as pd

# Local
from .constraint_matcher import TableFilter


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
    def apply_filter(self, table_filter: TableFilter, column_name: str):
        pass

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass


class DataFrameTable(Table):
    def __init__(self, df: pd.DataFrame):
        """The index of the dataframe is assumed to be the ID column.
        In other words, the ID column of a data frame must be set as its index
        before being passed into this constructor.
        """
        self.df = df

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

    def apply_filter(self, table_filter: TableFilter):
        return table_filter.filter_df_ids(self.df)


class SQLiteTable(Table):
    EVAL_PREFIX = "__eval__"
    TABLE_NAME = "sqlitetable"

    @staticmethod
    def _to_primitive(val):
        if isinstance(val, str) or isinstance(val, int) or isinstance(val, float):
            return val
        return f"{SQLiteTable.EVAL_PREFIX}{str(val)}"

    @staticmethod
    def _to_primitive_df(df):
        for col in df.columns:
            df[col] = df[col].apply(SQLiteTable._to_primitive)
        return df

    @staticmethod
    def _from_primitive(val):
        if isinstance(val, str) and val.startswith(SQLiteTable.EVAL_PREFIX):
            return eval(val[len(SQLiteTable.EVAL_PREFIX) :])
        return val

    @staticmethod
    def _from_primitive_df(df):
        for col in df.columns:
            df[col] = df[col].apply(SQLiteTable._from_primitive)
        return df

    def __init__(self, df: pd.DataFrame):
        # TODO: Reset index first?
        self.db_path = f"{uuid.uuid4()}.db"
        self.db_columns = df.columns
        self.db_size = len(df)
        if df.index.name:
            self.id_column = df.index.name
            df = df.reset_index()
        else:
            self.id_column = "__id__"
            while self.id_column in df.columns:
                self.id_column += "_"
            df[self.id_column] = range(len(df))

        df = SQLiteTable._to_primitive_df(df)

        # We don't save the db connection and instead create a new connection
        # each time to simplify serialization.
        con = sqlite3.connect(self.db_path)
        df.to_sql(name=SQLiteTable.TABLE_NAME, con=con)

    @property
    def columns(self) -> List[str]:
        return self.db_columns

    @property
    def size(self) -> int:
        return self.db_size

    @property
    def ids(self) -> List[int]:
        con = sqlite3.connect(self.db_path)
        return pd.read_sql(
            f"select {self.id_column} from {SQLiteTable.TABLE_NAME}", con
        )[self.id_column].apply(SQLiteTable._from_primitive)

    def field(self, row_id: int, column: str):
        con = sqlite3.connect(self.db_path)
        return SQLiteTable._from_primitive(
            pd.read_sql(
                f"select {column} from {SQLiteTable.TABLE_NAME} where {self.id_column}=={row_id}",
                con,
            )[column][0]
        )

    def row_as_dict(self, row_id: int) -> dict:
        con = sqlite3.connect(self.db_path)
        return SQLiteTable._from_primitive_df(
            pd.read_sql(
                f"select * from {SQLiteTable.TABLE_NAME} where {self.id_column}=={row_id}",
                con,
            )
        ).to_dict("records")[0]

    def range_rows_as_dicts(self, from_row_id: int, to_row_id: int) -> List[dict]:
        con = sqlite3.connect(self.db_path)
        return SQLiteTable._from_primitive_df(
            pd.read_sql(
                f"select * from {SQLiteTable.TABLE_NAME} where {self.id_column}>={from_row_id} and {self.id_column}<{to_row_id}",
                con,
            )
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
        print("DBPATH", self.db_path)

    def apply_filter(self, table_filter: TableFilter):
        return table_filter.filter_sql_ids(
            sqlite3.connect(self.db_path), SQLiteTable.TABLE_NAME
        )
