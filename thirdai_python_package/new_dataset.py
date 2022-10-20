import time

import numpy as np
import thirdai._thirdai.new_dataset
from thirdai._thirdai.new_dataset import *


def pandas_to_columnmap(df, dense_int_cols=set(), int_col_ranges={}):
    """
    Converts a pandas dataframe to a ColumnMap object. This method assumes that
    integer type columns are sparse. If you want to force an integer column to
    be dense, pass it in as an element of the dense_int_cols set. This method
    also assumes that any sparse columns have a range equal in size to the
    largest value in the column. This may not be always be the case (e.g.
    streaming or inference). You can explicitly pass in the actual range in the
    int_col_ranges dictionary with the key of the column name. Finally, note
    that the pandas array should have valid headers, as these will be the names
    of the column in the ColumnMap.
    """
    column_map = {}
    for column_name in df:
        column_np = df[column_name].values
        if np.issubdtype(column_np.dtype, np.floating) or column_name in dense_int_cols:
            column_map[column_name] = columns.NumpyDenseValueColumn(array=column_np)
        elif np.issubdtype(column_np.dtype, np.integer):
            dim = (
                int_col_ranges[column_name]
                if column_name in int_col_ranges
                else max(column_np) + 1
            )
            column_map[column_name] = columns.NumpySparseValueColumn(
                array=column_np, dim=dim
            )
        # If it is not an integer or float column, we assume it is a string
        else:
            column_map[column_name] = columns.StringColumn(array=column_np)

    return ColumnMap(column_map)


__all__ = ["pandas_to_columnmap"]
__all__.extend(dir(thirdai._thirdai.new_dataset))
