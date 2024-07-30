# Python
import shutil
import sqlite3
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Tuple

import dask.dataframe as dd

# Libraries
import pandas as pd

# Local
from .constraint_matcher import TableFilter
from .sql_helpers import df_to_sql, select_as_df


def df_with_index_name(df):
    index_name = df.index.name
    if not index_name:
        index_name = "__id__"
        while index_name in df.columns:
            index_name += "_"
        df.index.name = index_name
    return df


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


class DaskDataFrameTable(Table):
    def __init__(self, df: dd.DataFrame):
        self.df = df_with_index_name(df)
        self.row_id_to_dict = (
            {}
        )  # store row_id_to_dict before hand for quick retrieval in future
        index_name = self.df.index.name
        meta = pd.DataFrame({"data": pd.Series(dtype="object")})
        results = self.df.map_partitions(self._partition_to_dicts, meta=meta)
        row_id = 0  # We are maintaining a global row_id because each partition will have it's local index
        for batch in results.compute():
            for row_dict in batch:
                row_dict[index_name] = row_id
                self.row_id_to_dict[row_id] = row_dict
                row_id += 1

    @property
    def columns(self) -> List[str]:
        return [col for col in self.df.columns if col != self.df.index.name]

    @property
    def size(self) -> int:
        # For Dask, compute() is required to get the actual size
        return int(self.df.shape[0].compute())

    @property
    def ids(self) -> List[int]:
        # Dask requires computation to convert index to a list
        return self.df.index.compute().to_list()

    def field(self, row_id: int, column: str):
        # For Dask, use .compute() to get actual values
        return self.df.loc[row_id][column].compute()

    def row_as_dict(self, row_id: int) -> dict:
        return self.row_id_to_dict[row_id]

    def range_rows_as_dicts(self, from_row_id: int, to_row_id: int) -> List[dict]:
        return self.row_id_to_dict[from_row_id:to_row_id]

    def _partition_to_dicts(self, df_partition):
        dicts = []
        for row in df_partition.itertuples(index=True):
            row_dict = row._asdict()
            dicts.append(row_dict)
        return dicts

    def iter_rows_as_dicts(self) -> Generator[Tuple[int, dict], None, None]:
        for row_id, row_dict in self.row_id_to_dict.items():
            yield (row_id, row_dict)

    def apply_filter(self, table_filter: TableFilter):
        return table_filter.filter_df_ids(self.df)


class DataFrameTable(Table):
    def __init__(self, df: pd.DataFrame):
        """The index of the dataframe is assumed to be the ID column.
        In other words, the ID column of a data frame must be set as its index
        before being passed into this constructor.
        """
        self.df = df_with_index_name(df)

    @property
    def columns(self) -> List[str]:
        # Excludes ID column
        return self.df.columns

    @property
    def size(self) -> int:
        return len(self.df)

    @property
    def ids(self) -> List[int]:
        return self.df.index.to_list()

    def field(self, row_id: int, column: str):
        if column == self.df.index.name:
            return row_id
        return self.df[column].loc[row_id]

    def row_as_dict(self, row_id: int) -> dict:
        row = self.df.loc[row_id].to_dict()
        row[self.df.index.name] = row_id
        return row

    def range_rows_as_dicts(self, from_row_id: int, to_row_id: int) -> List[dict]:
        return (
            self.df.loc[from_row_id:to_row_id].reset_index().to_dict(orient="records")
        )

    def iter_rows_as_dicts(self) -> Generator[Tuple[int, dict], None, None]:
        for row in self.df.itertuples(index=True):
            row_id = row.Index
            row_dict = row._asdict()
            row_dict[self.df.index.name] = row_id
            yield (row_id, row_dict)

    def apply_filter(self, table_filter: TableFilter):
        return table_filter.filter_df_ids(self.df)


class SQLiteTable(Table):
    EVAL_PREFIX = "__eval__"
    TABLE_NAME = "sqlitetable"

    def __init__(self, df: pd.DataFrame):
        # TODO: Reset index first?
        self.db_path = Path(f"{uuid.uuid4()}.db").resolve()
        self.db_columns = df.columns
        self.db_size = len(df)

        # We don't save the db connection and instead create a new connection
        # each time to simplify serialization.
        df = df_with_index_name(df)
        self.id_column = df.index.name
        self.sql_table = df_to_sql(self.db_path, df, SQLiteTable.TABLE_NAME)

    @property
    def columns(self) -> List[str]:
        # Excludes ID column
        return self.db_columns

    @property
    def size(self) -> int:
        return self.db_size

    @property
    def ids(self) -> List[int]:
        return select_as_df(
            db_path=self.db_path,
            table=self.sql_table.c[self.id_column],
        )[self.id_column]

    def field(self, row_id: int, column: str):
        return select_as_df(
            db_path=self.db_path,
            table=self.sql_table.c[column],
            constraints=[self.sql_table.c[self.id_column] == row_id],
        )[column][0]

    def row_as_dict(self, row_id: int) -> dict:
        return select_as_df(
            db_path=self.db_path,
            table=self.sql_table,
            constraints=[self.sql_table.c[self.id_column] == row_id],
        ).to_dict("records")[0]

    def range_rows_as_dicts(self, from_row_id: int, to_row_id: int) -> List[dict]:
        return select_as_df(
            db_path=self.db_path,
            table=self.sql_table,
            constraints=[
                self.sql_table.c[self.id_column] >= from_row_id,
                self.sql_table.c[self.id_column] < to_row_id,
            ],
        ).to_dict("records")

    def select_with_constraint(self, column, value) -> List[dict]:
        return select_as_df(
            db_path=self.db_path,
            table=self.sql_table,
            constraints=[
                self.sql_table.c[column] == value,
            ],
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

    def apply_filter(self, table_filter: TableFilter):
        return table_filter.filter_sql_ids(
            sqlite3.connect(self.db_path), SQLiteTable.TABLE_NAME, self.id_column
        )
