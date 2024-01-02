import numpy as np
import pytest
from thirdai import smx

pytestmark = [pytest.mark.unit]


def run_dense_linear_test(x_np, N):
    K = x_np.shape[-1]

    w_np = np.random.rand(N, K).astype(np.float32)
    b_np = np.random.rand(N).astype(np.float32)

    y_np = np.matmul(x_np, w_np.T) + b_np

    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)
    w = smx.Variable(smx.from_numpy(w_np), requires_grad=True)
    b = smx.Variable(smx.from_numpy(b_np), requires_grad=True)

    y = smx.linear(x, w, b)

    assert np.allclose(y.tensor.numpy(), y_np)
    assert y.tensor.ndim == len(x_np.shape)

    y_grad_np = np.random.rand(*y_np.shape).astype(np.float32)
    y.backward(smx.from_numpy(y_grad_np))

    x_grad_np = np.matmul(y_grad_np, w_np)
    y_grad_np = y_grad_np.reshape(-1, y_grad_np.shape[-1])
    w_grad_np = np.matmul(y_grad_np.T, x_np.reshape(-1, x_np.shape[-1]))
    b_grad_np = np.sum(y_grad_np, axis=0)

    assert np.allclose(x_grad_np, x.grad.numpy())
    assert np.allclose(w_grad_np, w.grad.numpy())
    assert np.allclose(b_grad_np, b.grad.numpy())


def test_dense_linear_2d():
    M, K, N = 20, 10, 30

    x_np = np.random.rand(M, K).astype(np.float32)

    run_dense_linear_test(x_np, N=N)


def test_dense_linear_4d():
    M_1, M_2, M_3, K, N = 4, 5, 6, 10, 30

    x_np = np.random.rand(M_1, M_2, M_3, K).astype(np.float32)

    run_dense_linear_test(x_np, N=N)


def test_sparse_linear():
    M, K, N, nonzeros = 20, 100, 30, 20

    x_np = np.random.rand(M, K).astype(np.float32)
    w_np = np.random.rand(N, K).astype(np.float32)
    b_np = np.random.rand(N).astype(np.float32)

    x_indices = np.argsort(-x_np, axis=1)[:, :nonzeros].astype(np.uint32)
    x_values = x_np[np.arange(M)[:, None], x_indices]
    x_offsets = np.arange(0, M * nonzeros + 1, nonzeros, dtype=np.uint32)

    x = smx.CsrTensor(
        smx.from_numpy(x_offsets),
        smx.from_numpy(x_indices.reshape(-1)),
        smx.from_numpy(x_values.reshape(-1)),
        smx.Shape(M, K),
    )

    mask = np.zeros_like(x_np)
    mask[np.arange(M)[:, None], x_indices] = 1
    x_np = x_np * mask

    y_np = np.matmul(x_np, w_np.T) + b_np

    x = smx.Variable(x, requires_grad=True)
    w = smx.Variable(smx.from_numpy(w_np), requires_grad=True)
    b = smx.Variable(smx.from_numpy(b_np), requires_grad=True)

    y = smx.linear(x, w, b)

    assert np.allclose(y.tensor.numpy(), y_np)
    assert y.tensor.ndim == len(x_np.shape)

    y_grad_np = np.random.rand(*y_np.shape).astype(np.float32)
    y.backward(smx.from_numpy(y_grad_np))

    x_grad_np = np.matmul(y_grad_np, w_np)
    w_grad_np = np.matmul(y_grad_np.T, x_np.reshape(-1, x_np.shape[-1]))
    b_grad_np = np.sum(y_grad_np, axis=0)

    assert np.allclose(w_grad_np, w.grad.numpy())
    assert np.allclose(b_grad_np, b.grad.numpy())

    sparse_x_grad = x_grad_np[np.arange(M)[:, None], x_indices].reshape(-1)
    assert np.allclose(x.grad.row_offsets.numpy(), x_offsets)
    assert np.allclose(x.grad.col_indices.numpy(), x_indices.reshape(-1))
    assert np.allclose(x.grad.col_values.numpy(), sparse_x_grad)
