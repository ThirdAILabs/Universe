from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from thirdai._thirdai.data import ColumnMap, columns


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
            column_map[column_name] = columns.DecimalColumn(data=column_np)
        elif np.issubdtype(column_np.dtype, np.integer):
            dim = int_col_dims[column_name] if column_name in int_col_dims else None
            column_map[column_name] = columns.TokenColumn(data=column_np, dim=dim)
        elif _is_string_column(column_np):
            column_map[column_name] = columns.StringColumn(data=column_np)
        else:
            raise ValueError(
                f"All columns must be either an integer, float, or string type, but column {column_name} was none of these types."
            )

    return ColumnMap(column_map)
