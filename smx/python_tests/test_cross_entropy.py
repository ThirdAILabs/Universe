import numpy as np
import pytest
from thirdai import smx

pytestmark = [pytest.mark.unit]


def softmax_np(x):
    y = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return y / np.sum(y, axis=-1, keepdims=True)


def test_cross_entropy():
    M, K, N = 20, 10, 30
    x_np = np.random.rand(M, K, N).astype(np.float32)
    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)

    y_np = softmax_np(x_np)

    labels_np = np.random.randint(0, N, size=(M, K), dtype=np.uint32)
    labels = smx.from_numpy(labels_np)

    dense_labels = np.zeros((M, K, N), dtype=np.float32)
    dense_labels.reshape((-1, N))[np.arange(M * K), labels_np.reshape(-1)] = 1

    loss_np = np.sum(-np.log(y_np) * dense_labels) / (M * K)

    loss = smx.cross_entropy(x, labels)

    assert np.isclose(loss.tensor.scalar(), loss_np)

    loss.backward()

    x_grad_np = (dense_labels - y_np) / (M * K)

    assert np.allclose(x.grad.numpy(), x_grad_np)
