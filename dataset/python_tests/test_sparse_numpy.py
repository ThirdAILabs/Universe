import numpy as np
import pytest
from thirdai import dataset

pytestmark = [pytest.mark.unit]


def test_good_input():
    values = np.ones((100,), dtype="float32")
    indices = np.zeros((100,), dtype="uint32")
    offsets = np.arange(101, dtype="uint32")
    dataset.from_numpy((indices, values, offsets), batch_size=1)


def test_bad_indices():
    indices = np.zeros((100,), dtype="uint32")
    for len_offsets in [1, 10, 50, 200, 1000]:
        offsets = np.arange(len_offsets, dtype="uint32")
        values = np.ones((offsets[-1],), dtype="float32")
        with pytest.raises(
            ValueError,
            match=f".*the flattened indices array should be of length {offsets[-1]}, but it is actually of length 100",
        ):
            dataset.from_numpy((indices, values, offsets), batch_size=1)


def test_bad_values():
    values = np.ones((100,), dtype="float32")
    for len_offsets in [1, 10, 50, 200, 1000]:
        offsets = np.arange(len_offsets, dtype="uint32")
        indices = np.ones((offsets[-1],), dtype="uint32")
        with pytest.raises(
            ValueError,
            match=f".*the flattened values array should be of length {offsets[-1]}, but it is actually of length 100",
        ):
            dataset.from_numpy((indices, values, offsets), batch_size=1)


def test_empty_offsets():
    values = np.ones((100,), dtype="float32")
    indices = np.zeros((100,), dtype="uint32")
    offsets = np.array([], dtype="uint32")
    with pytest.raises(
        ValueError,
        match=f"Offsets array must be at least of size 1.*",
    ):
        dataset.from_numpy((indices, values, offsets), batch_size=1)
