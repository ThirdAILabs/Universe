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
def test_slice_does_not_copy():
    np_data = np.random.rand(10000, 1000)
    bolt_data = new_dataset.from_np(np_data)

    # This will go out of memory on any reasonable CI machine if we are doing
    # copies for each slice causing the test to fail (since
    # 4B bytes/entry * (10K * 1K) entries / slice * 100K slices ~= 4TB

    slices = []
    for _ in range(100000):
        slices.append(bolt_data[1:-1])

    for slice in slices:
        assert len(slice) == 10000 - 2
