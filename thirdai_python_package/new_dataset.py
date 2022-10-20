import time

import numpy as np
import thirdai._thirdai.new_dataset
from thirdai._thirdai.new_dataset import *


# df is a pandas dataframe, but we don't type it as such to avoid a package
# dependency on pandas.
# This method assumes that integer type columns are sparse. If you want to
# force those to be dense, pass them in as elements of the dense_int_cols set.
# This method also assumes that any sparse columns have a range equal in size
# to the largest value in the column. If this is not the case, pass in the
# actual range in the int_col_ranges dictionary with the key of the column name.
# Note also the pandas array should be named
def pandas_to_columnmap(df, dense_int_cols=set(), int_col_ranges={}):
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
