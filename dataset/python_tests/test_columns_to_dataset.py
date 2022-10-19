import numpy as np
import pytest
from dataset_utils import dense_bolt_dataset_to_numpy, sparse_bolt_dataset_to_numpy
from thirdai import dataset

pytestmark = [pytest.mark.unit]


def get_integer_value_column(n_rows, dim):
    column_np = np.random.randint(low=0, high=dim, size=(n_rows, 1))
    column = dataset.columns.NumpyIntegerValueColumn(array=column_np, dim=dim)
    return column, column_np


def get_float_value_column(n_rows):
    column_np = np.random.rand(n_rows, 1)
    column = dataset.columns.NumpyFloatValueColumn(array=column_np)
    return column, column_np


def get_integer_array_column(n_rows, dim, num_nonzeros):
    column_np = np.random.randint(low=0, high=dim, size=(n_rows, num_nonzeros))
    column = dataset.columns.NumpyIntegerArrayColumn(array=column_np, dim=dim)
    return column, column_np


def get_float_array_column(n_rows, dim):
    column_np = np.random.rand(n_rows, dim)
    column = dataset.columns.NumpyFloatArrayColumn(array=column_np)
    return column, column_np


def test_dense_columns_to_dataset():
    n_rows = 100

    column1, column1_np = get_float_value_column(n_rows)
    column2, column2_np = get_float_array_column(n_rows, dim=7)

    columns = dataset.ColumnMap({"column1": column1, "column2": column2})

    featurized_vectors = dense_bolt_dataset_to_numpy(
        columns.convert_to_dataset(["column1", "column2"], batch_size=21)
    )

    concatenated_columns = np.concatenate(
        [column1_np, column2_np], axis=1, dtype=np.float32
    )

    assert np.array_equal(featurized_vectors, concatenated_columns)


def test_sparse_columns_to_dataset():
    n_rows = 100

    column1_dim = 10
    column1, column1_np = get_integer_value_column(n_rows, dim=column1_dim)

    column2_dim = 20
    column2_nonzeros = 7
    column2, column2_np = get_integer_array_column(
        n_rows, dim=column2_dim, num_nonzeros=column2_nonzeros
    )

    columns = dataset.ColumnMap({"column1": column1, "column2": column2})

    indices, values = sparse_bolt_dataset_to_numpy(
        columns.convert_to_dataset(["column1", "column2"], batch_size=7)
    )

    concatenated_indices = np.concatenate(
        [column1_np, column2_np + column1_dim], axis=1
    )

    assert np.array_equal(indices, concatenated_indices)

    # Plus 1 is for column 1.
    assert np.array_equal(values, np.ones(shape=(n_rows, 1 + column2_nonzeros)))


def test_dense_sparse_columns_to_dataset():
    n_rows = 100

    column1, column1_np = get_float_value_column(n_rows)

    column2_dim = 20
    column2_nonzeros = 7
    column2, column2_np = get_integer_array_column(
        n_rows, dim=column2_dim, num_nonzeros=column2_nonzeros
    )

    columns = dataset.ColumnMap({"column1": column1, "column2": column2})

    indices, values = sparse_bolt_dataset_to_numpy(
        columns.convert_to_dataset(["column1", "column2"], batch_size=24)
    )

    concatenated_indices = np.concatenate(
        # Column 1 will have the indices with value 0, and we must offset the indices of column 2
        [np.zeros(shape=(n_rows, 1)), column2_np + 1],
        axis=1,
    )
    assert np.array_equal(indices, concatenated_indices)

    concatenated_values = np.concatenate(
        [column1_np, np.ones_like(column2_np)], axis=1, dtype=np.float32
    )
    assert np.array_equal(values, concatenated_values)


def get_dense_indices(n_rows, n_cols):
    return np.array([np.arange(n_cols) for _ in range(n_rows)])


def test_multiple_sparse_dense_columns_to_dataset():
    n_rows = 100

    column1_dim = 13
    column1, column1_np = get_float_array_column(n_rows, dim=column1_dim)

    column2, column2_np = get_float_value_column(n_rows)

    column3_dim = 10
    column3, column3_np = get_integer_value_column(n_rows, dim=column3_dim)

    column4_dim = 40
    column4_nonzeros = 8
    column4, column4_np = get_integer_array_column(
        n_rows, dim=column4_dim, num_nonzeros=column4_nonzeros
    )

    columns = dataset.ColumnMap(
        {"column1": column1, "column2": column2, "column3": column3, "column4": column4}
    )

    indices, values = sparse_bolt_dataset_to_numpy(
        columns.convert_to_dataset(
            ["column1", "column3", "column2", "column4"], batch_size=13
        )
    )

    concatenated_indices = np.concatenate(
        [
            get_dense_indices(n_rows, column1_dim),
            column3_np + column1_dim,
            get_dense_indices(n_rows, 1) + column1_dim + column3_dim,
            column4_np + column1_dim + 1 + column3_dim,  # The +1 is the column 2 dim
        ],
        axis=1,
    )
    assert np.array_equal(indices, concatenated_indices)

    concatenated_values = np.concatenate(
        [column1_np, np.ones_like(column3_np), column2_np, np.ones_like(column4_np)],
        axis=1,
        dtype=np.float32,
    )
    assert np.array_equal(values, concatenated_values)
