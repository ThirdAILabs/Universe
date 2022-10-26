from io import StringIO

import pandas as pd
import pytest
from thirdai import new_dataset

pytestmark = [pytest.mark.unit]


def test_basic_pandas_to_columnmap():
    TESTDATA = StringIO(
        """col1;col2;col3
            "lorem";4.4;99
            "ipsum";4.5;200
            "dolor";4.7;65"""
    )
    df = pd.read_csv(TESTDATA, sep=";")

    column_map = new_dataset.pandas_to_columnmap(df)

    assert isinstance(column_map["col1"], new_dataset.columns.StringColumn)
    assert isinstance(column_map["col2"], new_dataset.columns.NumpyDenseValueColumn)
    assert isinstance(column_map["col3"], new_dataset.columns.NumpySparseValueColumn)

    assert column_map["col1"].dimension_info() == None
    assert column_map["col2"].dimension_info().dim == 1
    assert column_map["col3"].dimension_info() == None


def test_pandas_to_columnmap_int_cols():
    TESTDATA = StringIO(
        """col1;col2;col3
            1;1;1
            5;5;5
            10;10;10"""
    )
    df = pd.read_csv(TESTDATA, sep=";")

    column_map = new_dataset.pandas_to_columnmap(
        df, dense_int_cols={"col2"}, int_col_dims={"col3": 20}
    )

    assert isinstance(column_map["col1"], new_dataset.columns.NumpySparseValueColumn)
    assert isinstance(column_map["col2"], new_dataset.columns.NumpyDenseValueColumn)
    assert isinstance(column_map["col3"], new_dataset.columns.NumpySparseValueColumn)

    assert column_map["col1"].dimension_info() == None
    assert column_map["col2"].dimension_info().dim == 1
    assert column_map["col3"].dimension_info().dim == 20


def test_pandas_bad_col():
    df = pd.DataFrame({"col1": [1, 2], "col2": [(1, 2), (3, 4)]})
    with pytest.raises(
        ValueError,
        match="All columns must be either an integer, float, or string type, but column col2 was none of these types.",
    ):
        new_dataset.pandas_to_columnmap(df)
