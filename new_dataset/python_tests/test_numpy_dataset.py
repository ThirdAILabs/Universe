import numpy as np
import pytest
from thirdai import data


def rows_equal(bolt_dataset_1, bolt_dataset_2):
    for r1, r2 in zip(bolt_dataset_1, bolt_dataset_2):
        if str(r1) != str(r2):
            return False
    return True


@pytest.mark.unit
def test_basic_dense_numpy():
    num_rows = 10
    num_cols = 10
    np_data = np.full((num_rows, num_cols), 1)
    bolt_data = data.from_np(np_data)

    row_count = 0
    for row in bolt_data:
        row_count += 1
        assert str(row) == str([1 for _ in range(num_cols)])
    assert row_count == num_rows


@pytest.mark.unit
def test_simple_slice():
    np_data = np.random.rand(10, 10)
    bolt_data = data.from_np(np_data)
    slice = bolt_data[2:4]

    for i in range(2, 4):
        assert str(slice[i - 2]) == str(bolt_data[i])


@pytest.mark.unit
def test_shuffle_works():

    np.random.seed(42)
    np_data = np.random.rand(10, 10).astype("float32")
    np.random.shuffle(np_data)
    bolt_data_shuffled_in_numpy = data.from_np(np_data)

    np.random.seed(42)
    np_data = np.random.rand(10, 10).astype("float32")
    bolt_data_shuffled_in_dataset = data.from_np(np_data)
    np.random.shuffle(bolt_data_shuffled_in_dataset)

    assert rows_equal(bolt_data_shuffled_in_numpy, bolt_data_shuffled_in_dataset)


@pytest.mark.unit
def test_slice_is_a_view():
    """
    Tests that a slice of a dataset is a view to the same shared underlying
    memory as the original dataset by ensuring that modifications to the slice
    are reflected in the original dataset and vice versa.
    """
    np_data = np.random.rand(40, 10)

    bolt_data = data.from_np(np_data)
    first_half = bolt_data[0:20]

    # This sets both the first half of bolt_data and first_half equal to the
    # second half of bolt_data if views work correctly, which we then check in
    # the next two loops.
    first_half[0:10] = bolt_data[20:30]
    bolt_data[10:20] = bolt_data[30:40]

    assert rows_equal(first_half, bolt_data[0:20])

    second_half = bolt_data[20:40]
    assert rows_equal(first_half, second_half)


@pytest.mark.unit
def test_bad_slices():
    np_data = np.random.rand(40, 10)

    bolt_data = data.from_np(np_data)

    with pytest.raises(
        ValueError,
        match="Slices must have positive size, but found start index 30 and end index 30",
    ):
        bolt_data[30:30]

    with pytest.raises(
        ValueError,
        match="Slices must have positive size, but found start index 20 and end index 10",
    ):
        bolt_data[20:10]

    with pytest.raises(
        ValueError,
        match="Slices must have positive size, but found start index 30 and end index 25",
    ):
        bolt_data[-10:-15]

    with pytest.raises(
        ValueError,
        match="Slices must have positive size, but found start index 40 and end index 40",
    ):
        bolt_data[500:510]

    with pytest.raises(ValueError, match="Dataset slices must have step size 1"):
        bolt_data[20:10:2]


@pytest.mark.unit
def test_dataset_copy():
    np_data = np.random.rand(40, 10)
    bolt_data = data.from_np(np_data)
    bolt_data_copy = bolt_data.copy()

    bolt_data[0] = bolt_data[1]

    assert not rows_equal(bolt_data, bolt_data_copy)
