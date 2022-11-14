import tempfile

import pytest
from thirdai.data import _CATEGORICAL_DELIMITERS, semantic_type_inference


@pytest.mark.parametrize("delimiter", _CATEGORICAL_DELIMITERS)
def test_basic_type_inference(delimiter):
    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        tmp.write(
            f"""col1,col2,col3,col4,col5,col6
                lorem,2,3.0,label1{delimiter}label2{delimiter}label3,How vexingly quick daft zebras jump!,2021-02-01
                ipsum,5,6,label4,"Sphinx of black quartz, judge my vow.",2022-02-01
                dolor,8,9,label5{delimiter}label6,The quick brown fox jumps over the lazy dog,2023-02-01
            """
        )
        tmp.flush()

        inferred_types = semantic_type_inference(tmp.name)

        assert inferred_types["col1"]["type"] == "categorical"
        assert inferred_types["col2"]["type"] == "categorical"
        assert inferred_types["col3"]["type"] == "numerical"
        assert inferred_types["col4"]["type"] == "multi-categorical"
        assert inferred_types["col4"]["delimiter"] == delimiter
        assert inferred_types["col5"]["type"] == "text"
        assert inferred_types["col6"]["type"] == "datetime"


def test_short_file_throws_error():
    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        tmp.write(
            """col1,
            1
            2
            """
        )
        tmp.flush()

        with pytest.raises(
            ValueError, match=".*must have at least 3 rows, but we found only 2 rows."
        ):
            semantic_type_inference(tmp.name)
