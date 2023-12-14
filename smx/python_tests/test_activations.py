import numpy as np
import pytest
from thirdai import smx

pytestmark = [pytest.mark.unit]


def run_test(smx_fn, np_fn, np_grad_fn):
    x_np = np.random.rand(10, 20, 30).astype(np.float32)
    y_np = np_fn(x_np)

    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)
    y = smx_fn(x)

    assert np.allclose(y.tensor.numpy(), y_np)

    y_grad_np = np.random.rand(*y_np.shape).astype(np.float32)
    x_grad_np = np_grad_fn(y_np, y_grad_np)

    y_grad = smx.from_numpy(y_grad_np)
    y.backward(y_grad)

    assert np.allclose(x.grad.numpy(), x_grad_np)


def test_relu():
    run_test(
        smx_fn=smx.relu,
        np_fn=lambda x: x.clip(0),
        np_grad_fn=lambda y, y_grad: np.where(y > 0, y_grad, 0),
    )


def test_tanh():
    run_test(
        smx_fn=smx.tanh,
        np_fn=lambda x: np.tanh(x),
        np_grad_fn=lambda y, y_grad: (1 - np.square(y)) * y_grad,
    )


def test_sigmoid():
    run_test(
        smx_fn=smx.sigmoid,
        np_fn=lambda x: 1 / (1 + np.exp(-x)),
        np_grad_fn=lambda y, y_grad: (y - np.square(y)) * y_grad,
    )


def softmax_np(x):
    y = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return y / np.sum(y, axis=-1, keepdims=True)


def softmax_grad_np(y, y_grad):
    gy = y * y_grad
    jacobian = gy - y * np.sum(gy, axis=-1, keepdims=True)
    return jacobian * y_grad


def test_softmax():
    run_test(
        smx_fn=smx.softmax,
        np_fn=softmax_np,
        np_grad_fn=softmax_grad_np,
    )
