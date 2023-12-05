from thirdai import smx
import numpy as np
import pytest
import re


pytestmark = [pytest.mark.unit]


def test_shape():
    shape = smx.Shape(1, 2, 3, 4)

    assert len(shape) == 4
    assert str(shape) == "(1, 2, 3, 4)"

    contents = []
    for i, x in enumerate(shape):
        assert (i + 1) == x
        contents.append(x)

    assert contents == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "dtype", [(np.float32, smx.Dtype.f32), (np.uint32, smx.Dtype.u32)]
)
def test_from_numpy(dtype):
    x = np.arange(24).astype(dtype[0]).reshape(2, 3, 4)

    tensor = smx.from_numpy(x)

    assert tensor.ndim == 3
    assert list(tensor.shape) == [2, 3, 4]
    assert list(tensor.strides) == [12, 4, 1]

    assert tensor.dtype == dtype[1]


def test_to_numpy_f32():
    x = np.random.rand(5, 2, 74, 26, 3).astype(np.float32)
    tensor = smx.from_numpy(x)

    assert np.array_equal(x, tensor.numpy())


def test_to_numpy_u32():
    x = np.random.randint(0, 10000, (9, 28, 24, 1, 30), dtype=np.uint32)
    tensor = smx.from_numpy(x)

    assert np.array_equal(x, tensor.numpy())


def test_reshape():
    x = np.random.randint(0, 10000, (2, 3, 4, 5, 6), dtype=np.uint32)

    tensor = smx.reshape(smx.from_numpy(x), new_shape=smx.Shape(6, 4, 30))

    assert np.array_equal(x.reshape(6, 4, 30), tensor.numpy())

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot reshape tensor with shape (2, 3, 4, 5, 6) to shape (7, 4, 30)."
        ),
    ):
        smx.reshape(smx.from_numpy(x), smx.Shape(7, 4, 30))


@pytest.mark.parametrize("ndim", [2, 3, 4, 5, 6, 7, 8])
def test_transpose(ndim):
    shape = list(np.random.randint(3, 8, size=ndim))

    perm = list(range(ndim))
    while perm == list(range(ndim)):
        np.random.shuffle(perm)
    assert not np.array_equal(perm, np.arange(ndim))

    x = np.random.rand(*shape).astype(np.float32)

    tensor = smx.transpose(smx.from_numpy(x), perm)

    assert np.array_equal(np.transpose(x, perm), tensor.numpy())
