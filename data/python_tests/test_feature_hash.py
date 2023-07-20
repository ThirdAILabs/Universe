import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def test_same_indices_in_column():
    columns = data.ColumnMap(
        {"col": data.columns.TokenArrayColumn([[1, 1], [2, 2], [3, 3]])}
    )

    transformation = data.transformations.FeatureHash(
        columns=["col"],
        output_indices="indices",
        output_values="values",
        dim=100000,
    )

    columns = transformation(columns)

    rows = columns["indices"].data()

    assert len(rows) == 3

    # Check that the same token gets hashed to the same location in the same column.
    for row in rows:
        assert len(set(row)) == 1


def test_different_indices_in_column():
    columns = data.ColumnMap(
        {"col": data.columns.TokenArrayColumn([[1, 2], [2, 3], [3, 4]])}
    )

    transformation = data.transformations.FeatureHash(
        columns=["col"],
        output_indices="indices",
        output_values="values",
        dim=100000,
    )

    columns = transformation(columns)

    rows = columns["indices"].data()

    assert len(rows) == 3

    # Check that different tokens get hashed to different locations in the same column.
    for row in rows:
        assert len(set(row)) == 2


def test_same_indices_in_different_columns():
    columns = data.ColumnMap(
        {
            "col1": data.columns.TokenColumn([1, 2, 3]),
            "col2": data.columns.TokenColumn([1, 2, 3]),
        }
    )

    transformation = data.transformations.FeatureHash(
        columns=["col1", "col2"],
        output_indices="indices",
        output_values="values",
        dim=100000,
    )

    columns = transformation(columns)

    rows = columns["indices"].data()

    assert len(rows) == 3

    # Check that the same token in different columns gets hashed to different locations.
    for row in rows:
        assert len(set(row)) == 2


def test_decimal_values_preserved():
    columns = data.ColumnMap(
        {
            "col1": data.columns.DecimalColumn([1.0, 2.0, 3.0]),
            "col2": data.columns.DecimalArrayColumn(
                [[10.0, 20.0], [11.0, 22.0], [12.0, 24.0]]
            ),
        }
    )

    transformation = data.transformations.FeatureHash(
        columns=["col1", "col2"],
        output_indices="indices",
        output_values="values",
        dim=100000,
    )

    columns = transformation(columns)

    rows = columns["values"].data()
    expected_rows = [[1.0, 10.0, 20.0], [2.0, 11.0, 22.0], [3.0, 12.0, 24.0]]

    assert len(rows) == len(expected_rows)

    # Check that all of the decimal values are present.
    for row, expected_row in zip(rows, expected_rows):
        assert set(row) == set(expected_row)

    # Check that the indices hashed to are unique.
    for row in columns["indices"].data():
        assert len(set(row)) == 3
