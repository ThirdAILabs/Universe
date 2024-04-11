import numpy as np
import pytest
from thirdai import smx


@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
def test_concat(dim):
    shape = (3, 4, 5, 6, 7)
    a_np = np.random.rand(*shape).astype(np.float32)
    b_np = np.random.rand(*shape).astype(np.float32)
    c_np = np.random.rand(*shape).astype(np.float32)

    a = smx.Variable(smx.from_numpy(a_np), requires_grad=True)
    b = smx.Variable(smx.from_numpy(b_np), requires_grad=True)
    c = smx.Variable(smx.from_numpy(c_np), requires_grad=True)

    out = smx.concat([a, b, c], dim=dim)
    out_np = np.concatenate([a_np, b_np, c_np], axis=dim)

    assert np.array_equal(out.tensor.numpy(), out_np)

    grad_np = np.random.rand(*out_np.shape).astype(np.float32)

    grads_np = np.split(
        grad_np, [a_np.shape[dim], a_np.shape[dim] + b_np.shape[dim]], axis=dim
    )

    out.backward(smx.from_numpy(grad_np))

    assert np.array_equal(a.grad.numpy(), grads_np[0])
    assert np.array_equal(b.grad.numpy(), grads_np[1])
    assert np.array_equal(c.grad.numpy(), grads_np[2])
