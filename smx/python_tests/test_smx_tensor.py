import re

import numpy as np
import pytest
from thirdai import smx

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


def test_reshape_tensor():
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


def test_reshape_autograd():
    x_np = np.random.randint(0, 10000, (2, 3, 4, 5, 6), dtype=np.uint32)

    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)
    y = smx.reshape(x, new_shape=smx.Shape(6, 4, 30))

    assert np.array_equal(x_np.reshape(6, 4, 30), y.tensor.numpy())

    y_grad_np = np.random.randint(0, 10000, (6, 4, 30), dtype=np.uint32)

    y.backward(smx.from_numpy(y_grad_np))

    assert np.array_equal(y_grad_np.reshape(2, 3, 4, 5, 6), x.grad.numpy())


@pytest.mark.parametrize("ndim", [2, 3, 4, 5, 6, 7, 8])
def test_transpose_tensor(ndim):
    shape = list(np.random.randint(3, 8, size=ndim))

    perm = list(range(ndim))
    while perm == list(range(ndim)):
        np.random.shuffle(perm)
    assert not np.array_equal(perm, list(range(ndim)))

    x = np.random.rand(*shape).astype(np.float32)

    tensor = smx.transpose(smx.from_numpy(x), perm)

    assert np.array_equal(np.transpose(x, perm), tensor.numpy())


@pytest.mark.parametrize("ndim", [2, 3, 4, 5, 6, 7, 8])
def test_transpose_autograd(ndim):
    shape = list(np.random.randint(3, 8, size=ndim))

    perm = list(range(ndim))
    while perm == list(range(ndim)):
        np.random.shuffle(perm)
    assert not np.array_equal(perm, list(range(ndim)))

    x_np = np.random.rand(*shape).astype(np.float32)

    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)

    y = smx.transpose(x, perm)

    y_np = np.transpose(x_np, perm)

    assert np.array_equal(y_np, y.tensor.numpy())

    # Applying transpose backward on the output should return to the input.
    y.backward(smx.from_numpy(y_np))

    assert np.array_equal(x_np, x.grad.numpy())


def test_csr_tensor():
    offsets = [0, 3, 7, 9, 13]
    indices = [5, 3, 4, 7, 11, 19, 15, 6, 1, 16, 18, 0, 8]
    values = [1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4]

    tensor = smx.CsrTensor(offsets, indices, values, smx.Shape(4, 20))

    assert tensor.n_rows == 4
    assert tensor.n_dense_cols == 20
    assert np.array_equal(np.array(offsets), tensor.row_offsets.numpy())
    assert np.array_equal(np.array(indices), tensor.col_indices.numpy())
    assert np.array_equal(np.array(values), tensor.col_values.numpy())


def test_csr_invalid_offsets():
    bad_offsets = [
        [0, 3, 7, 9, 12],
        [1, 3, 7, 9, 13],
        [0, 3, 7, 9, 14],
        [0, 3, 7, 6, 14],
    ]
    indices = [5, 3, 4, 7, 11, 19, 15, 6, 1, 16, 18, 0, 8]
    values = [1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4]

    for offsets in bad_offsets:
        with pytest.raises(RuntimeError):
            smx.CsrTensor(offsets, indices, values, smx.Shape(4, 20))

    with pytest.raises(RuntimeError):
        smx.CsrTensor([0, 3, 7, 9, 13], indices, values, smx.Shape(3, 20))


def test_csr_invalid_indices():
    offsets = [0, 3, 7, 9, 13]
    indices = [5, 3, 4, 7, 11, 19, 15, 6, 1, 16, 18, 0, 8]
    bad_indices = [5, 3, 4, 7, 11, 19, 15, 60, 1, 16, 18, 0, 8]
    values = [1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4]

    with pytest.raises(RuntimeError):
        smx.CsrTensor(offsets, indices, [], smx.Shape(4, 20))

    with pytest.raises(RuntimeError):
        smx.CsrTensor(offsets, [], values, smx.Shape(4, 20))

    with pytest.raises(RuntimeError):
        smx.CsrTensor(offsets, bad_indices, values, smx.Shape(4, 20))


def test_tensor_indexing():
    array = np.random.rand(2, 3, 4, 2).astype(np.float32)
    tensor = smx.from_numpy(array)

    for i in range(array.shape[0]):
        assert np.array_equal(tensor[i].numpy(), array[i])
        for j in range(array.shape[1]):
            assert np.array_equal(tensor[i, j].numpy(), array[i, j])
            for k in range(array.shape[2]):
                assert np.array_equal(tensor[i, j, k].numpy(), array[i, j, k])
                for n in range(array.shape[3]):
                    assert tensor[i, j, k, n].scalar() == array[i, j, k, n]
