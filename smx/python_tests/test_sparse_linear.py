import numpy as np
import pytest
from thirdai import smx


def test_sparse_linear():
    batch_size, dim, input_dim, nonzeros = 2, 6, 4, 3
    x_np = np.random.rand(batch_size, input_dim).astype(np.float32)
    w_np = np.random.rand(dim, input_dim).astype(np.float32)
    b_np = np.random.rand(dim).astype(np.float32)

    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)
    w = smx.Variable(smx.from_numpy(w_np), requires_grad=True)
    b = smx.Variable(smx.from_numpy(b_np), requires_grad=True)

    y_np = np.matmul(x_np, w_np.T) + b_np

    sparse_linear = smx.SparseLinear(dim, input_dim, nonzeros / dim)
    sparse_linear.weight = w
    sparse_linear.bias = b

    y = sparse_linear([x])[0]

    assert tuple(y.tensor.shape) == (batch_size, dim)
    assert np.array_equal(
        y.tensor.row_offsets.numpy(),
        np.arange(
            start=0, stop=batch_size * nonzeros + 1, step=nonzeros, dtype=np.uint32
        ),
    )
    assert tuple(y.tensor.col_indices.shape) == (batch_size * nonzeros,)
    assert tuple(y.tensor.col_values.shape) == (batch_size * nonzeros,)

    indices = y.tensor.col_indices.numpy().reshape(batch_size, nonzeros)

    expected_values = y_np[np.arange(batch_size)[:, None], indices].reshape(-1)

    assert np.allclose(y.tensor.col_values.numpy(), expected_values)

    mask = np.zeros_like(y_np)
    mask[np.arange(batch_size)[:, None], indices] = 1

    y_grad_sparse_np = np.random.rand(batch_size, nonzeros).astype(np.float32)
    y_grad_dense_np = np.zeros((batch_size, dim), dtype=np.float32)
    y_grad_dense_np[np.arange(batch_size)[:, None], indices] = y_grad_sparse_np

    x_grad_np = np.matmul(y_grad_dense_np, w_np)
    w_grad_np = np.matmul(y_grad_dense_np.T, x_np)
    b_grad_np = np.sum(y_grad_dense_np, axis=0)

    y.backward(
        smx.CsrTensor(
            y.tensor.row_offsets,
            y.tensor.col_indices,
            smx.from_numpy(y_grad_sparse_np.reshape(-1)),
            y.tensor.shape,
        )
    )

    assert np.allclose(x.grad.numpy(), x_grad_np)
    assert np.allclose(w.grad.numpy(), w_grad_np)
    assert np.allclose(b.grad.numpy(), b_grad_np)
