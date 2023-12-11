import numpy as np
import pytest
from thirdai import smx


pytestmark = [pytest.mark.unit]


def run_test(smx_fn, np_fn, np_grad_fn):
    M, N = 20, 30

    x_np = np.random.rand(M, N).astype(np.float32)
    y_np = np_fn(x_np)

    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)
    y = smx_fn(x)

    assert np.allclose(y.tensor.numpy(), y_np)

    y_grad_np = np.random.rand(M, N).astype(np.float32)
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
