import numpy as np
import pytest
from thirdai import smx


pytestmark = [pytest.mark.unit]


def test_linear():
    M, K, N = 20, 10, 30

    x_np = np.random.rand(M, K).astype(np.float32)
    w_np = np.random.rand(N, K).astype(np.float32)
    b_np = np.random.rand(N).astype(np.float32)

    y_np = np.matmul(x_np, w_np.T) + b_np

    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)
    w = smx.Variable(smx.from_numpy(w_np), requires_grad=True)
    b = smx.Variable(smx.from_numpy(b_np), requires_grad=True)

    y = smx.linear(x, w, b)

    assert np.allclose(y.tensor.numpy(), y_np)

    y_grad_np = np.random.rand(M, N).astype(np.float32)
    y_grad = smx.from_numpy(y_grad_np)
    y.backward(y_grad)

    x_grad_np = np.matmul(y_grad_np, w_np)
    w_grad_np = np.matmul(y_grad_np.T, x_np)
    b_grad_np = np.sum(y_grad_np, axis=0)

    assert np.allclose(x_grad_np, x.grad.numpy())
    assert np.allclose(w_grad_np, w.grad.numpy())
    assert np.allclose(b_grad_np, b.grad.numpy())
