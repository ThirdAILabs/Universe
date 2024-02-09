import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def test_same_indices_in_column():
    ROWS = 100
    columns = data.ColumnMap(
        {"col": data.columns.TokenArrayColumn([[i, i] for i in range(ROWS)])}
    )

    transformation = data.transformations.FeatureHash(
        input_columns=["col"],
        output_indices_column="indices",
        output_values_column="values",
        hash_range=100000,
    )

    columns = transformation(columns)

    rows = columns["indices"].data()

    assert len(rows) == ROWS

    # Check that the same token gets hashed to the same location in the same column.
    for row in rows:
        assert len(set(row)) == 1


def test_different_indices_in_column():
    ROWS = 100
    COLS = 5
    col_data = [[i + j for j in range(COLS)] for i in range(ROWS)]
    columns = data.ColumnMap({"col": data.columns.TokenArrayColumn(col_data)})

    transformation = data.transformations.FeatureHash(
        input_columns=["col"],
        output_indices_column="indices",
        output_values_column="values",
        hash_range=100000,
    )

    columns = transformation(columns)

    rows = columns["indices"].data()

    assert len(rows) == ROWS

    # Check that different tokens get hashed to different locations in the same
    # column. Normally we could expect some hash collisions but the range is large
    # enough relative to the number of elements such that this doesn't happen. Also
    # the feature hashing is deterministic, so this will not be flaky.
    for row in rows:
        assert len(set(row)) == COLS

    # Check that the overlapping tokens between rows match
    for i in range(ROWS):
        if i < (ROWS - 1):
            assert len(set(rows[i]).intersection(set(rows[i + 1]))) == (COLS - 1)
        if i < (ROWS - 2):
            assert len(set(rows[i]).intersection(set(rows[i + 2]))) == (COLS - 2)


def test_same_indices_in_different_columns():
    ROWS = 100
    columns = data.ColumnMap(
        {
            "col1": data.columns.TokenColumn([i for i in range(ROWS)]),
            "col2": data.columns.TokenColumn([i for i in range(ROWS)]),
        }
    )

    transformation = data.transformations.FeatureHash(
        input_columns=["col1", "col2"],
        output_indices_column="indices",
        output_values_column="values",
        hash_range=100000,
    )

    columns = transformation(columns)

    rows = columns["indices"].data()

    assert len(rows) == ROWS

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
        input_columns=["col1", "col2"],
        output_indices_column="indices",
        output_values_column="values",
        hash_range=100000,
    )

    columns = transformation(columns)

    values = columns["values"].data()
    expected_rows = [[1.0, 10.0, 20.0], [2.0, 11.0, 22.0], [3.0, 12.0, 24.0]]

    assert len(values) == len(expected_rows)

    # Check that all of the decimal values are present.
    for row, expected_row in zip(values, expected_rows):
        assert set(row) == set(expected_row)

    # Check that the indices hashed to are unique, and that the indices are consistent.
    indices = columns["indices"].data()
    for row in indices:
        assert len(set(row)) == 3
        assert row == indices[0]


def test_tokens_and_decimals():
    ROWS = 100
    COLS = 5

    tokens = [[i + j for j in range(COLS)] for i in range(ROWS)]
    decimals = [[float(i + j + 10) ** 2 for j in range(COLS)] for i in range(ROWS)]

    columns = data.ColumnMap(
        {
            "col1": data.columns.TokenArrayColumn(tokens),
            "col2": data.columns.DecimalArrayColumn(decimals),
        }
    )

    transformation = data.transformations.FeatureHash(
        input_columns=["col1", "col2"],
        output_indices_column="indices",
        output_values_column="values",
        hash_range=100000,
    )

    columns = transformation(columns)

    assert len(columns) == ROWS

    indices = [set(x) for x in columns["indices"].data()]
    values = columns["values"].data()

    # Only the decimal indices should overlap
    decimal_indices = indices[0].intersection(indices[-1])
    assert len(decimal_indices) == COLS

    decimal_locs = [values[0].index(x) for x in decimals[0]]
    decimal_indices = set([columns["indices"][0][i] for i in decimal_locs])

    for i in range(ROWS):
        # Check that the row has the correct number of unique indices.
        assert len(indices[i]) == (COLS * 2)
        # Check that the row has the correct number of values
        assert len(values[i]) == (COLS * 2)
        # Check that the row has the correct number of unique values.
        assert set(values[i]) == set([1.0] + decimals[i])
        # Check that the indices of the decimal values are in the row, these should
        # be the same for every row.
        assert len(indices[i].intersection(decimal_indices)) == COLS

        # Check that the values that correspond to the indices of the decimal values
        # are the correct decimal values.
        decimals_in_row = set(values[i][decimal_loc] for decimal_loc in decimal_locs)
        assert decimals_in_row == set(decimals[i])

        # Check the intersection of the indices with the indices of the next row.
        # All except for 1 token should be the same between consecutive rows.
        if i < (ROWS - 1):
            assert len(indices[i].intersection(indices[i + 1])) == (COLS * 2 - 1)
        if i < (ROWS - 3):
            assert len(indices[i].intersection(indices[i + 3])) == (COLS * 2 - 3)


def test_feature_hash_serialization():
    ROWS = 100
    COLS = 5

    tokens = [[i + j for j in range(COLS)] for i in range(ROWS)]
    decimals = [[float(i + j + 10) ** 2 for j in range(COLS)] for i in range(ROWS)]

    columns = data.ColumnMap(
        {
            "col1": data.columns.TokenArrayColumn(tokens),
            "col2": data.columns.DecimalArrayColumn(decimals),
        }
    )

    transformation = data.transformations.FeatureHash(
        input_columns=["col1", "col2"],
        output_indices_column="indices",
        output_values_column="values",
        hash_range=100000,
    )

    transformation_copy = data.transformations.deserialize(transformation.serialize())

    output1 = transformation(columns)
    output2 = transformation_copy(columns)

    assert output1["indices"].data() == output2["indices"].data()
    assert output1["values"].data() == output2["values"].data()
