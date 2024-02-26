import numpy as np
import pytest
from thirdai import smx

pytestmark = [pytest.mark.unit]


def softmax_np(x):
    y = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return y / np.sum(y, axis=-1, keepdims=True)


def test_cross_entropy_dense_single_label():
    M, K, N = 20, 10, 30
    x_np = np.random.rand(M, K, N).astype(np.float32)
    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)

    y_np = softmax_np(x_np)

    labels_np = np.random.randint(0, N, size=(M, K), dtype=np.uint32)
    labels = smx.from_numpy(labels_np)

    dense_labels = np.zeros((M, K, N), dtype=np.float32)
    dense_labels.reshape((-1, N))[np.arange(M * K), labels_np.reshape(-1)] = 1

    loss_np = np.sum(-np.log(y_np) * dense_labels) / (M * K)

    loss = smx.cross_entropy(x, smx.Variable(labels, requires_grad=False))

    assert np.isclose(loss.tensor.scalar(), loss_np)

    loss.backward()

    x_grad_np = (dense_labels - y_np) / (M * K)

    assert np.allclose(x.grad.numpy(), x_grad_np)


def test_cross_entropy_sparse_single_label():
    offsets = [0, 4, 7, 12, 14]
    indices = [6, 3, 4, 8, 0, 1, 5, 3, 7, 2, 9, 6, 0, 8]
    x1 = np.array([1.0, 3.0, 0.5, -2.5])
    x2 = np.array([1.5, 0.25, 3.5])
    x3 = np.array([2.75, 1.25, 0.5, 1.75, -0.5])
    x4 = np.array([2.5, 1.0])
    x = smx.CsrTensor(
        row_offsets=offsets,
        col_indices=indices,
        col_values=list(np.concatenate([x1, x2, x3, x4])),
        dense_shape=smx.Shape(4, 10),
    )

    labels = smx.from_numpy(np.array([4, 1, 6, 0], dtype=np.uint32))

    x = smx.Variable(x, requires_grad=True)
    loss = smx.cross_entropy(x, smx.Variable(labels, requires_grad=False))

    y1 = softmax_np(x1)
    y2 = softmax_np(x2)
    y3 = softmax_np(x3)
    y4 = softmax_np(x4)
    expected_loss = (-np.log(y1[2]) - np.log(y2[1]) - np.log(y3[4]) - np.log(y4[0])) / 4

    assert np.isclose(loss.tensor.scalar(), expected_loss)
    loss.backward()

    assert np.array_equal(
        x.grad.row_offsets.numpy(), np.array(offsets, dtype=np.uint32)
    )
    assert np.array_equal(
        x.grad.col_indices.numpy(), np.array(indices, dtype=np.uint32)
    )

    expected_grad = -np.concatenate([y1, y2, y3, y4])
    expected_grad[[2, 5, 11, 12]] += 1.0
    expected_grad /= 4
    assert np.allclose(x.grad.col_values.numpy(), expected_grad)


def test_cross_entropy_dense_multi_label():
    x_np = np.random.uniform(-2, 2, size=(4, 10))
    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)

    l1, l1_v = [0, 7, 4], [0.5, 0.4, 0.1]
    l2, l2_v = [8, 1], [0.4, 0.7]
    l3, l3_v = [4, 9, 0], [1.0, 1.5, 0.5]
    l4, l4_v = [6], [1.0]

    y_np = softmax_np(x_np)

    l_dense = np.zeros_like(x_np)
    l_dense[0, l1] = l1_v
    l_dense[1, l2] = l2_v
    l_dense[2, l3] = l3_v
    l_dense[3, l4] = l4_v

    labels = smx.CsrTensor(
        [0, 3, 5, 8, 9], l1 + l2 + l3 + l4, l1_v + l2_v + l3_v + l4_v, smx.Shape(4, 10)
    )

    loss = smx.cross_entropy(x, smx.Variable(labels, requires_grad=False))

    assert np.isclose(loss.tensor.scalar(), np.sum(-l_dense * np.log(y_np)) / 4)

    loss.backward()

    assert np.allclose(
        x.grad.numpy(), (l_dense - np.sum(l_dense, axis=-1, keepdims=True) * y_np) / 4
    )


def test_cross_entropy_sparse_multi_label():
    offsets = [0, 4, 7, 12, 14]
    x1, x1_v = [6, 3, 4, 8], [1.0, 3.0, 0.5, -2.5]
    x2, x2_v = [0, 1, 5], [1.5, 0.25, 3.5]
    x3, x3_v = [3, 7, 2, 9, 6], [2.75, 1.25, 0.5, 1.75, -0.5]
    x4, x4_v = [0, 8], [2.5, 1.0]

    l1, l1_v, m1 = [6, 8, 3], [0.5, 0.4, 0.1], [0.5, 0.1, 0.0, 0.4]
    l2, l2_v, m2 = [0, 5], [0.4, 0.7], [0.4, 0.0, 0.7]
    l3, l3_v, m3 = [3, 9, 2], [1.0, 1.5, 0.5], [1.0, 0.0, 0.5, 1.5, 0.0]
    l4, l4_v, m4 = [0], [1.0], [1.0, 0.0]

    x = smx.CsrTensor(
        offsets,
        x1 + x2 + x3 + x4,
        x1_v + x2_v + x3_v + x4_v,
        smx.Shape(4, 10),
    )
    x = smx.Variable(x, requires_grad=True)
    labels = smx.CsrTensor(
        [0, 3, 5, 8, 9],
        l1 + l2 + l3 + l4,
        l1_v + l2_v + l3_v + l4_v,
        smx.Shape(4, 10),
    )

    loss = smx.cross_entropy(x, smx.Variable(labels, requires_grad=False))

    full_acts = np.concatenate(
        [softmax_np(np.array(xi)) for xi in [x1_v, x2_v, x3_v, x4_v]]
    )
    expected_loss = np.sum(-np.array(m1 + m2 + m3 + m4) * np.log(full_acts)) / 4
    assert np.isclose(loss.tensor.scalar(), expected_loss)

    loss.backward()

    expected_grad = np.concatenate(
        [
            sum(li) * softmax_np(np.array(xi))
            for xi, li in [(x1_v, l1_v), (x2_v, l2_v), (x3_v, l3_v), (x4_v, l4_v)]
        ]
    )

    expected_grad = (np.array(m1 + m2 + m3 + m4) - expected_grad) / 4

    assert np.array_equal(x.grad.row_offsets.numpy(), offsets)
    assert np.array_equal(
        x.grad.col_indices.numpy(), np.array(x1 + x2 + x3 + x4, dtype=np.uint32)
    )

    assert np.allclose(x.grad.col_values.numpy(), expected_grad)
