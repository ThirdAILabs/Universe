import pytest
from dataset_utils import get_random_str_column
from thirdai import data


@pytest.mark.unit
def test_string_concat():
    ROWS = 1000

    columns = {f"col_{i}": get_random_str_column(ROWS) for i in range(10)}
    input_columns = list(columns.keys())
    columns = data.ColumnMap(columns)

    transformation = data.transformations.StringConcat(
        input_columns, "output", seperator="#"
    )

    columns = transformation(columns)

    for i in range(ROWS):
        expected = "#".join([columns[col][i] for col in input_columns])
        assert columns["output"][i] == expected
