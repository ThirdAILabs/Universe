from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import thirdai._thirdai.data
from thirdai._thirdai.data import *
import thirdai._thirdai.bolt


class ColumnMapGenerator(ABC):
    @abstractmethod
    def next() -> Optional[ColumnMap]:
        pass

    @abstractmethod
    def restart() -> None:
        pass


def _is_string_column(column):
    return all([isinstance(s, str) for s in column])


def pandas_to_columnmap(df, dense_int_cols=set(), int_col_dims={}):
    """
    Converts a pandas dataframe to a ColumnMap object. This method assumes that
    integer type columns are sparse. If you want to force an integer column to
    be dense, pass the name of the column as an element of the dense_int_cols
    set. This method will also assume that integer columns are
    non-concatenatable (i.e. they have None for the dim), but you can explicitly
    pass in the actual range in the int_col_dims dictionary with the key of the
    column name . Finally, note that the pandas array should have valid headers,
    as these will be the names of the column in the ColumnMap.
    """
    column_map = {}
    for column_name in df:
        column_np = df[column_name].to_numpy()
        if np.issubdtype(column_np.dtype, np.floating) or column_name in dense_int_cols:
            column_map[column_name] = columns.NumpyDenseValueColumn(array=column_np)
        elif np.issubdtype(column_np.dtype, np.integer):
            dim = int_col_dims[column_name] if column_name in int_col_dims else None
            column_map[column_name] = columns.NumpySparseValueColumn(
                array=column_np, dim=dim
            )
        elif _is_string_column(column_np):
            column_map[column_name] = columns.StringColumn(array=column_np)
        else:
            raise ValueError(
                f"All columns must be either an integer, float, or string type, but column {column_name} was none of these types."
            )

    return ColumnMap(column_map)


def get_metadata(filename, n_rows=1e6):
    column_types = infer_types(filename)

    df = pd.read_csv(filename, n_rows=n_rows)

    udt_column_types = {}

    for col_name in df.columns:
        if col_name not in column_types:
            raise ValueError("Dataframe contains columns not in column_type map.")
        col_type = column_types[col_name]

        if col_type == "text":
            udt_column_types[col_name] = bolt.types.text()
        elif col_type == "categorical":
            udt_column_types[col_name] = bolt.types.categorical()
        elif col_type == "multicategorical":
            udt_column_types[col_name] = bolt.types.text()
        elif col_type == "numeric":
            min_val = df[col_name].min()
            max_val = df[col_name].max()
            udt_column_types[col_name] = bolt.types.numerical(range=(min_val, max_val))
        elif col_type == "timestamp":
            udt_column_types[col_name] = bolt.types.date()
        else:
            raise ValueError("Received invalid column type. Supports 'text', 'categorical', 'multicategorical', 'numeric', and 'timestamp'.")
    
    return udt_column_types


__all__ = ["ColumnMapGenerator", "pandas_to_columnmap"]
__all__.extend(dir(thirdai._thirdai.data))
