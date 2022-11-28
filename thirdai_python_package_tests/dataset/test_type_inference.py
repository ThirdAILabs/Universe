import tempfile

import pytest
from py_dataset_utils import create_csv_and_pqt_files
from thirdai.data import _CATEGORICAL_DELIMITERS, semantic_type_inference

pytestmark = [pytest.mark.unit]

CSV_FILENAME = "data.csv"
PARQUET_FILENAME = "data.pqt"
FILENAMES = [CSV_FILENAME, PARQUET_FILENAME]


@pytest.mark.parametrize("delimiter", _CATEGORICAL_DELIMITERS)
def test_basic_type_inference(delimiter):
    create_csv_and_pqt_files(
        CSV_FILENAME,
        PARQUET_FILENAME,
        f"""col1,col2,col3,col4,col5,col6\n
            lorem,2,3.0,label1{delimiter}label2{delimiter}label3,How vexingly quick daft zebras jump!,2021-02-01\n
            ipsum,5,6,label4,"Sphinx of black quartz, judge my vow.",2022-02-01\n
            dolor,8,9,label5{delimiter}label6,The quick brown fox jumps over the lazy dog,2023-02-01\n
        """,
    )

    for filename in FILENAMES:
        inferred_types = semantic_type_inference(filename)

        assert inferred_types["col1"]["type"] == "categorical"
        assert inferred_types["col2"]["type"] == "categorical"
        assert inferred_types["col3"]["type"] == "numerical"
        assert inferred_types["col4"]["type"] == "multi-categorical"
        assert inferred_types["col4"]["delimiter"] == delimiter
        assert inferred_types["col5"]["type"] == "text"
        assert inferred_types["col6"]["type"] == "datetime"


# Tests that we can still do type inference with missing values. This is the
# same test as test_basic_type_inference except every column has a missing value
# on row number col_index % 3.
@pytest.mark.parametrize("delimiter", _CATEGORICAL_DELIMITERS)
def test_type_inference_missing_vals(delimiter):
    create_csv_and_pqt_files(
        CSV_FILENAME,
        PARQUET_FILENAME,
        f"""col1,col2,col3,col4,col5,col6\n
                ,2,3.0,,How vexingly quick daft zebras jump!,2021-02-01\n
                ipsum,,6,label1{delimiter}label2{delimiter}label3,,2022-02-01\n
                dolor,8,,label5{delimiter}label6,The quick brown fox jumps over the lazy dog,\n
            """,
    )

    # TODO (david, josh) get this test to work with parquet too
    inferred_types = semantic_type_inference(CSV_FILENAME)

    assert inferred_types["col1"]["type"] == "categorical"
    assert inferred_types["col2"]["type"] == "categorical"
    assert inferred_types["col3"]["type"] == "numerical"
    assert inferred_types["col4"]["type"] == "multi-categorical"
    assert inferred_types["col4"]["delimiter"] == delimiter
    assert inferred_types["col5"]["type"] == "text"
    assert inferred_types["col6"]["type"] == "datetime"


def test_short_file_throws_error():
    create_csv_and_pqt_files(
        CSV_FILENAME,
        PARQUET_FILENAME,
        """col1,\n
            1\n
            2\n
            """,
    )

    for filename in FILENAMES:
        with pytest.raises(
            ValueError, match=".*must have at least 3 rows, but we found only 2 rows."
        ):
            semantic_type_inference(filename)


def test_short_file_with_missing_vals_throws_error():
    create_csv_and_pqt_files(
        CSV_FILENAME,
        PARQUET_FILENAME,
        """col1,col2\n
            1,2\n
            ,\n
            ,\n
            ,\n
            ,\n
            """,
    )

    for filename in FILENAMES:
        with pytest.raises(
            ValueError,
            match=".*has less than less than 2 non-missing values so we cannot do type inference",
        ):
            semantic_type_inference(filename)
