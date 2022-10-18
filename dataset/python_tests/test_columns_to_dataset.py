import numpy as np
import pytest
from thirdai import dataset

pytestmark = [pytest.mark.unit]


def dense_vectors_to_numpy(vectors):
    return np.array([v.numpy() for v in vectors])


def sparse_vectors_to_numpy(vectors):
    indices_list = []
    values_list = []
    for vec in vectors:
        (i, v) = vec.numpy()
        indices_list.append(i)
        values_list.append(v)

    indices = np.array(indices_list)
    values = np.array(values_list)
    return (indices, values)


def test_simple_dense_columns():
    n_rows = 100

    column1_np = np.random.rand(n_rows, 1)
    column2_np = np.random.rand(n_rows, 7)

    column1 = dataset.columns.NumpyFloatValueColumn(array=column1_np)
    column2 = dataset.columns.NumpyFloatArrayColumn(array=column2_np)

    columns = dataset.ColumnMap({"column1": column1, "column2": column2})

    featurized_vectors = dense_vectors_to_numpy(
        columns.convert_to_dataset(["column1", "column2"])
    )

    concatenated_columns = np.concatenate([column1_np, column2_np], axis=1, dtype=np.float32)

    assert np.array_equal(featurized_vectors, concatenated_columns)


def test_simple_sparse_columns():
    n_rows = 100

    column1_dim = 10
    column1_len = 1
    column1_np = np.random.randint(low=0, high=column1_dim, size=(n_rows, column1_len))

    column2_dim = 20
    column2_len = 7
    column2_np = np.random.randint(low=0, high=column2_dim, size=(n_rows, column2_len))

    column1 = dataset.columns.NumpyIntegerValueColumn(array=column1_np, dim=column1_dim)
    column2 = dataset.columns.NumpyIntegerArrayColumn(array=column2_np, dim=column2_dim)

    columns = dataset.ColumnMap({"column1": column1, "column2": column2})

    indices, values = sparse_vectors_to_numpy(columns.convert_to_dataset(["column1", "column2"]))

    concatenated_indices = np.concatenate(
        [column1_np, column2_np + column1_dim], axis=1
    )

    assert np.array_equal(indices, concatenated_indices)

    assert np.array_equal(values, np.ones(shape=(n_rows, column1_len + column2_len)))


def test_simple_dense_sparse_columns():
    n_rows = 100

    column1_dim = 1
    column1_np = np.random.rand(n_rows, column1_dim)

    column2_dim = 20
    column2_len = 7
    column2_np = np.random.randint(low=0, high=column2_dim, size=(n_rows, column2_len))

    column1 = dataset.columns.NumpyFloatValueColumn(array=column1_np)
    column2 = dataset.columns.NumpyIntegerArrayColumn(array=column2_np, dim=column2_dim)

    columns = dataset.ColumnMap({"column1": column1, "column2": column2})

    indices, values = sparse_vectors_to_numpy(columns.convert_to_dataset(["column1", "column2"]))

    concatenated_indices = np.concatenate(
        [np.zeros(shape=(n_rows, column1_dim)), column2_np + column1_dim], axis=1
    )
    assert np.array_equal(indices, concatenated_indices)

    concatenated_values = np.concatenate(
        [column1_np, np.ones_like(column2_np)], axis=1, dtype=np.float32
    )
    assert np.array_equal(values, concatenated_values)


def get_dense_indices(n_rows, n_cols):
    return np.array([np.arange(n_cols) for _ in range(n_rows)])


def test_multiple_sparse_dense_columns():
    n_rows = 100

    column1_dim = 13
    column1_np = np.random.rand(n_rows, column1_dim)

    column2_dim = 1
    column2_np = np.random.rand(n_rows, column2_dim)

    column3_dim = 10
    column3_len = 1
    column3_np = np.random.randint(low=0, high=column3_dim, size=(n_rows, column3_len))

    column4_dim = 40
    column4_len = 8
    column4_np = np.random.randint(low=0, high=column4_dim, size=(n_rows, column4_len))

    column1 = dataset.columns.NumpyFloatArrayColumn(array=column1_np)
    column2 = dataset.columns.NumpyFloatValueColumn(array=column2_np)
    column3 = dataset.columns.NumpyIntegerValueColumn(array=column3_np, dim=column3_dim)
    column4 = dataset.columns.NumpyIntegerArrayColumn(array=column4_np, dim=column4_dim)

    columns = dataset.ColumnMap(
        {"column1": column1, "column2": column2, "column3": column3, "column4": column4}
    )

    indices, values = sparse_vectors_to_numpy(columns.convert_to_dataset(["column1", "column3", "column2", "column4"]))

    concatenated_indices = np.concatenate(
        [
            get_dense_indices(n_rows, column1_dim),
            column3_np + column1_dim,
            get_dense_indices(n_rows, column2_dim) + column1_dim + column3_dim,
            column4_np + column1_dim + column2_dim + column3_dim,
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
