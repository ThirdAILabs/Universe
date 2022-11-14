import tempfile

import pytest
from thirdai.data import _CATEGORICAL_DELIMITERS, get_udt_col_types

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("delimiter", _CATEGORICAL_DELIMITERS)
def test_get_udt_columns(delimiter):
    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        tmp.write(
            f"""col1,col2,col3,col4,col5,col6
                lorem,2,3.0,label1{delimiter}label2{delimiter}label3,How vexingly quick daft zebras jump!,2021-02-01
                ipsum,5,6,label4,"Sphinx of black quartz, judge my vow.",2022-02-01
                dolor,8,9,label5{delimiter}label6,The quick brown fox jumps over the lazy dog,2023-02-01
            """
        )
        tmp.flush()

        udt_types = get_udt_col_types(tmp.name)

        print(type(udt_types["col1"]))

        assert udt_types["col1"] == "categorical"
        assert udt_types["col2"] == "categorical"
        assert udt_types["col3"] == "numerical"
        assert udt_types["col3"].range == (3, 9)
        assert udt_types["col4"] == "multi-categorical"
        assert udt_types["col4"].delimiter == delimiter
        assert udt_types["col5"] == "text"
        assert udt_types["col6"] == "datetime"
