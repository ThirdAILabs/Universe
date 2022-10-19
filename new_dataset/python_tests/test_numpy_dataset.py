import numpy as np
import pytest
from thirdai import new_dataset


@pytest.mark.unit
def test_basic_dense_numpy():
    num_rows = 10
    num_cols = 10
    np_data = np.full((num_rows, num_cols), 1)
    bolt_data = new_dataset.from_np(np_data)

    row_count = 0
    for row in bolt_data:
        row_count += 1
        assert str(row) == str([1 for _ in range(num_cols)])
    assert row_count == num_rows


@pytest.mark.unit
def test_simple_slice():
    np_data = np.random.rand(10, 10)
    bolt_data = new_dataset.from_np(np_data)
    slice = bolt_data[2:4]

    for i in range(2, 4):
        assert str(slice[i - 2]) == str(bolt_data[i])


@pytest.mark.unit
def test_shuffle_works():

    np.random.seed(42)
    np_data = np.random.rand(10, 10).astype("float32")
    np.random.shuffle(np_data)
    bolt_data_shuffled_in_numpy = new_dataset.from_np(np_data)

    np.random.seed(42)
    np_data = np.random.rand(10, 10).astype("float32")
    bolt_data_shuffled_in_dataset = new_dataset.from_np(np_data)
    np.random.shuffle(bolt_data_shuffled_in_dataset)

    for r1, r2 in zip(bolt_data_shuffled_in_numpy, bolt_data_shuffled_in_dataset):
        assert str(r1) == str(r2)


@pytest.mark.unit
def test_slice_is_a_view():
    """
    Tests that a slice of a dataset is a view to the same shared underlying
    memory as the original dataset by ensuring that modifications to the slice
    are reflected in the original dataset and vice versa.
    """
    np_data = np.random.rand(40, 10)

    bolt_data = new_dataset.from_np(np_data)
    first_half = bolt_data[0:20]

    # This sets both the first half of bolt_data and first_half equal to the
    # second half of bolt_data if views work correctly, which we then check in
    # the next two loops.
    first_half[0:10] = bolt_data[20:30]
    bolt_data[10:20] = bolt_data[30:40]

    for r1, r2 in zip(first_half, bolt_data[0:20]):
        assert str(r1) == str(r2)

    second_half = bolt_data[20:40]
    for r1, r2 in zip(first_half, second_half):
        assert str(r1) == str(r2)
