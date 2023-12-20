import numpy as np
import pytest
from thirdai import smx

pytestmark = [pytest.mark.unit]


def run_dense_test(smx_fn, np_fn, np_grad_fn, atol=1e-8):
    x_np = np.random.uniform(-10, 10, size=(10, 20, 30)).astype(np.float32)
    y_np = np_fn(x_np)

    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)
    y = smx_fn(x)

    assert np.allclose(y.tensor.numpy(), y_np)

    y_grad_np = np.random.rand(*y_np.shape).astype(np.float32)
    x_grad_np = np_grad_fn(y_np, y_grad_np)

    y_grad = smx.from_numpy(y_grad_np)
    y.backward(y_grad)

    assert np.allclose(x.grad.numpy(), x_grad_np, atol=atol)


def run_sparse_test(smx_fn, np_fn, np_grad_fn, atol=1e-8):
    batch_size, dim = 20, 10000
    offsets = np.array(
        [0] + list(np.cumsum(np.random.randint(3, 20, size=batch_size))),
        dtype=np.uint32,
    )
    indices = np.random.randint(0, dim, size=offsets[-1], dtype=np.uint32)
    in_values = np.random.uniform(-10, 10, size=offsets[-1]).astype(np.float32)

    out_values = np_fn(offsets, in_values)

    out_values_grad = np.random.rand(offsets[-1]).astype(np.float32)
    in_values_grad = np_grad_fn(offsets, out_values, out_values_grad)

    x = smx.Variable(
        smx.CsrTensor(offsets, indices, in_values, smx.Shape(batch_size, dim)),
        requires_grad=True,
    )

    y = smx_fn(x)

    assert np.array_equal(y.tensor.row_offsets.numpy(), offsets)
    assert np.array_equal(y.tensor.col_indices.numpy(), indices)
    assert np.allclose(y.tensor.col_values.numpy(), out_values)

    y_grad = smx.CsrTensor(
        offsets, indices, out_values_grad, smx.Shape(batch_size, dim)
    )

    y.backward(y_grad)

    assert np.array_equal(x.grad.row_offsets.numpy(), offsets)
    assert np.array_equal(x.grad.col_indices.numpy(), indices)
    assert np.allclose(x.grad.col_values.numpy(), in_values_grad, atol=atol)


def test_relu():
    np_fn = lambda x: x.clip(0)
    np_grad_fn = lambda y, y_grad: np.where(y > 0, y_grad, 0)

    run_dense_test(smx_fn=smx.relu, np_fn=np_fn, np_grad_fn=np_grad_fn)
    run_sparse_test(
        smx_fn=smx.relu,
        np_fn=lambda offsets, x: np_fn(x),
        np_grad_fn=lambda offsets, y, y_grad: np_grad_fn(y, y_grad),
    )


def test_tanh():
    np_fn = lambda x: np.tanh(x)
    np_grad_fn = lambda y, y_grad: (1 - np.square(y)) * y_grad

    run_dense_test(smx_fn=smx.tanh, np_fn=np_fn, np_grad_fn=np_grad_fn, atol=1e-6)
    run_sparse_test(
        smx_fn=smx.tanh,
        np_fn=lambda offsets, x: np_fn(x),
        np_grad_fn=lambda offsets, y, y_grad: np_grad_fn(y, y_grad),
        atol=1e-6,
    )


def test_sigmoid():
    np_fn = lambda x: 1 / (1 + np.exp(-x))
    np_grad_fn = lambda y, y_grad: (y - np.square(y)) * y_grad

    run_dense_test(smx_fn=smx.sigmoid, np_fn=np_fn, np_grad_fn=np_grad_fn, atol=1e-6)
    run_sparse_test(
        smx_fn=smx.sigmoid,
        np_fn=lambda offsets, x: np_fn(x),
        np_grad_fn=lambda offsets, y, y_grad: np_grad_fn(y, y_grad),
        atol=1e-6,
    )


def softmax_np(x):
    y = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return y / np.sum(y, axis=-1, keepdims=True)


def softmax_grad_np(y, y_grad):
    gy = y * y_grad
    jacobian = gy - y * np.sum(gy, axis=-1, keepdims=True)
    return jacobian * y_grad


def softmax_sparse_np(offsets, x):
    out = []
    for i in range(len(offsets) - 1):
        x_i = x[offsets[i] : offsets[i + 1]]
        out.append(softmax_np(x_i))
    return np.concatenate(out, axis=0)


def softmax_sparse_grad_np(offsets, y, y_grad):
    out = []
    for i in range(len(offsets) - 1):
        y_i = y[offsets[i] : offsets[i + 1]]
        y_g_i = y_grad[offsets[i] : offsets[i + 1]]
        out.append(softmax_grad_np(y_i, y_g_i))
    return np.concatenate(out, axis=0)


def test_softmax():
    run_dense_test(
        smx_fn=smx.softmax, np_fn=softmax_np, np_grad_fn=softmax_grad_np, atol=1e-6
    )
    run_sparse_test(
        smx_fn=smx.softmax,
        np_fn=softmax_sparse_np,
        np_grad_fn=softmax_sparse_grad_np,
        atol=1e-6,
    )
