import os
import tempfile

import pandas as pd
import pytest
from py_dataset_utils import create_csv_and_pqt_files
from thirdai.data import _CATEGORICAL_DELIMITERS, get_udt_col_types

pytestmark = [pytest.mark.unit]


def verify_col_types(col_types, delimiter):
    assert str(col_types["col1"]) == '{"type": "categorical"}'
    assert str(col_types["col2"]) == '{"type": "categorical"}'
    assert (
        str(col_types["col3"])
        == '{"type": "numerical", "range": [3, 9], "granularity": "m"}'
    )
    assert (
        str(col_types["col4"])
        == '{"type": "categorical", "delimiter": "' + delimiter + '"}'
    )
    assert str(col_types["col5"]) == '{"type": "text"}'
    assert str(col_types["col6"]) == '{"type": "date"}'


@pytest.mark.parametrize("delimiter", _CATEGORICAL_DELIMITERS)
def test_get_udt_columns(delimiter):
    csv_filename = "data.csv"
    pqt_filename = "data.pqt"

    create_csv_and_pqt_files(
        csv_filename,
        pqt_filename,
        f"""col1,col2,col3,col4,col5,col6\n
            lorem,2,3.0,label1{delimiter}label2{delimiter}label3,How vexingly quick daft zebras jump!,2021-02-01\n
            ipsum,5,6,label4,"Sphinx of black quartz, judge my vow.",2022-02-01\n
            dolor,8,9,label5{delimiter}label6,The quick brown fox jumps over the lazy dog,2023-02-01\n
        """,
    )

    udt_types_csv = get_udt_col_types(csv_filename)
    verify_col_types(udt_types_csv, delimiter)

    udt_types_pqt = get_udt_col_types(pqt_filename)
    verify_col_types(udt_types_pqt, delimiter)

    os.remove(csv_filename)
    os.remove(pqt_filename)
