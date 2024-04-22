from thirdai import data
import pytest


@pytest.mark.unit
def test_parse_svm():
    columns = data.ColumnMap(
        {
            "input": data.columns.StringColumn(
                [
                    "1,2,3 55:0.875 24:1.75 60:-0.125",
                    "10  81:1.5\t",
                    "20,21 1:2.25\t11:22.5",
                ]
            )
        }
    )

    columns = data.transformations.ParseSvm(
        input_col="input",
        indices_col="indices",
        values_col="values",
        labels_col="labels",
        indices_dim=100,
        labels_dim=30,
    )(columns)

    assert columns["indices"].data() == [[55, 24, 60], [81], [1, 11]]
    assert columns["values"].data() == [[0.875, 1.75, -0.125], [1.5], [2.25, 22.5]]
    assert columns["labels"].data() == [[1, 2, 3], [10], [20, 21]]
