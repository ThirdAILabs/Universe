import random
import string

import numpy as np
import pytest
from thirdai import data

pytestmark = [pytest.mark.unit]


def test_token_column():
    ROWS = 10000
    tokens = list(range(ROWS))

    columns = [
        data.columns.TokenColumn(tokens, dim=ROWS),
        data.columns.TokenColumn(np.array(tokens), dim=ROWS),
    ]

    for column in columns:
        assert column.data() == tokens
        assert column.dim() == ROWS
        assert len(column) == ROWS

        for i, token in enumerate(tokens):
            assert column[i] == token


def test_decimal_column():
    ROWS = 10000
    decimals = [random.random() for _ in range(ROWS)]

    columns = [
        data.columns.DecimalColumn(decimals),
        data.columns.DecimalColumn(np.array(decimals)),
    ]

    for column in columns:
        assert np.allclose(column.data(), decimals)
        assert column.dim() == 1
        assert len(column) == ROWS

        for i, decimal in enumerate(decimals):
            assert np.allclose([column[i]], [decimal])


def test_string_column():
    ROWS = 10000
    strings = [
        "".join(random.choices(string.ascii_lowercase, k=10)) for _ in range(ROWS)
    ]

    column = data.columns.StringColumn(strings)

    assert column.data() == strings
    assert column.dim() == None
    assert len(column) == ROWS

    for i, s in enumerate(strings):
        assert column[i] == s


def test_timestamp_column():
    ROWS = 10000
    timestamps = list(range(ROWS))

    column = data.columns.TimestampColumn(timestamps)

    assert column.data() == timestamps
    assert column.dim() == None
    assert len(column) == ROWS
    for i, t in enumerate(timestamps):
        assert column[i] == t


def test_token_array_column():
    ROWS = 10000
    COLS = 100

    tokens = [[i * COLS + j for j in range(COLS)] for i in range(ROWS)]

    columns = [
        data.columns.TokenArrayColumn(tokens, dim=ROWS * COLS),
        data.columns.TokenArrayColumn(np.array(tokens), dim=ROWS * COLS),
    ]

    for column in columns:
        assert column.data() == tokens
        assert column.dim() == ROWS * COLS
        assert len(column) == ROWS

        for i, row in enumerate(tokens):
            assert np.array_equal(column[i], np.array(row))


def test_token_array_column_non_uniform():
    ROWS = 100

    tokens = [list(range(i)) for i in range(ROWS)]

    column = data.columns.TokenArrayColumn(tokens, dim=ROWS * ROWS)

    assert column.data() == tokens
    assert column.dim() == ROWS * ROWS
    assert len(column) == ROWS

    for i, row in enumerate(tokens):
        assert np.array_equal(column[i], np.array(row))


def test_decimal_array_column():
    ROWS = 10000
    COLS = 100

    decimals = [[random.random() for _ in range(COLS)] for _ in range(ROWS)]

    columns = [
        data.columns.DecimalArrayColumn(decimals, dim=COLS),
        data.columns.DecimalArrayColumn(np.array(decimals), dim=COLS),
    ]

    for column in columns:
        assert np.allclose(column.data(), decimals)
        assert column.dim() == COLS
        assert len(column) == ROWS

        for i, row in enumerate(decimals):
            assert np.allclose(column[i], np.array(row))


def test_decimal_array_column_non_uniform():
    ROWS = 100

    decimals = [list(map(float, range(i))) for i in range(ROWS)]

    column = data.columns.DecimalArrayColumn(decimals)

    assert column.data() == decimals
    assert column.dim() == None
    assert len(column) == ROWS

    for i, row in enumerate(decimals):
        assert np.allclose(column[i], np.array(row))


def test_token_columns_without_dimension():
    assert data.columns.TokenColumn([1, 2, 3]).dim() == None

    assert data.columns.TokenArrayColumn([[1], [2], [3]]).dim() == None


def test_decimal_array_column_without_dimension():
    assert data.columns.DecimalArrayColumn([[1.0], [2.0], [3.0]]).dim() == None


def test_token_columns_with_invalid_token():
    with pytest.raises(
        ValueError, match="Invalid index 10 for TokenColumn with dimension 5."
    ):
        data.columns.TokenColumn([10], dim=5)

    with pytest.raises(
        ValueError, match="Invalid index 10 for TokenArrayColumn with dimension 5."
    ):
        data.columns.TokenArrayColumn([[10]], dim=5)


@pytest.mark.parametrize(
    "column_type", [data.columns.TokenColumn, data.columns.DecimalColumn]
)
def test_pass_invalid_dimensions_to_value_column(column_type):
    with pytest.raises(
        ValueError, match="Expected 1D array when creating ValueColumn."
    ):
        column_type(np.array([[10, 20]]))


@pytest.mark.parametrize(
    "column_type", [data.columns.TokenArrayColumn, data.columns.DecimalArrayColumn]
)
def test_pass_invalid_dimensions_to_array_column(column_type):
    with pytest.raises(
        ValueError, match="Expected 2D array when creating ArrayColumn."
    ):
        column_type(np.array([10]))

    with pytest.raises(
        ValueError, match="Expected 2D array when creating ArrayColumn."
    ):
        column_type(np.array([[[1, 2], [2, 2]]])).data()
