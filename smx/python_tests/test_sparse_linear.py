import numpy as np
import pytest
from thirdai import smx

pytestmark = [pytest.mark.unit]

BATCH_SIZE, DIM, INPUT_DIM, NONZEROS = 8, 80, 7, 10


def run_sparse_linear_test(labels, indices_check, exact_param_grad_check, bias):
    x_np = np.random.rand(BATCH_SIZE, INPUT_DIM).astype(np.float32)
    w_np = np.random.rand(DIM, INPUT_DIM).astype(np.float32)
    b_np = np.random.rand(DIM).astype(np.float32) if bias else None

    x = smx.Variable(smx.from_numpy(x_np), requires_grad=True)
    w = smx.Variable(smx.from_numpy(w_np), requires_grad=True)
    b = smx.Variable(smx.from_numpy(b_np), requires_grad=True) if bias else None

    y_np = np.matmul(x_np, w_np.T)
    if bias:
        y_np += b_np

    sparse_linear = smx.SparseLinear(DIM, INPUT_DIM, NONZEROS / DIM, bias=bias)
    sparse_linear.weight = w
    sparse_linear.bias = b

    y = sparse_linear(x, labels)

    assert tuple(y.tensor.shape) == (BATCH_SIZE, DIM)
    assert np.array_equal(
        y.tensor.row_offsets.numpy(),
        np.arange(
            start=0, stop=BATCH_SIZE * NONZEROS + 1, step=NONZEROS, dtype=np.uint32
        ),
    )
    assert tuple(y.tensor.col_indices.shape) == (BATCH_SIZE * NONZEROS,)
    assert tuple(y.tensor.col_values.shape) == (BATCH_SIZE * NONZEROS,)

    indices = y.tensor.col_indices.numpy().reshape(BATCH_SIZE, NONZEROS)
    for row in indices:
        assert len(set(row)) == NONZEROS
    indices_check(labels, indices)

    expected_values = y_np[np.arange(BATCH_SIZE)[:, None], indices].reshape(-1)

    assert np.allclose(y.tensor.col_values.numpy(), expected_values)

    mask = np.zeros_like(y_np)
    mask[np.arange(BATCH_SIZE)[:, None], indices] = 1

    y_grad_sparse_np = np.random.rand(BATCH_SIZE, NONZEROS).astype(np.float32)
    y_grad_dense_np = np.zeros((BATCH_SIZE, DIM), dtype=np.float32)
    y_grad_dense_np[np.arange(BATCH_SIZE)[:, None], indices] = y_grad_sparse_np

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

    # The data race on the weight and bias gradients means that this sometimes
    # fails. On one test the labels are passed in such that the selected neurons
    # are disjoint, in which case the exact gradients can be checked entirely.
    if exact_param_grad_check:
        assert np.allclose(w.grad.numpy(), w_grad_np)
        if bias:
            assert np.allclose(b.grad.numpy(), b_grad_np)
    else:
        assert np.mean(np.isclose(w.grad.numpy(), w_grad_np)) >= 0.8
        if bias:
            assert np.mean(np.isclose(b.grad.numpy(), b_grad_np)) >= 0.8


@pytest.mark.parametrize("bias", [True, False])
def test_sparse_linear_no_label(bias):
    run_sparse_linear_test(
        labels=None,
        indices_check=lambda a, b: None,
        exact_param_grad_check=True,
        bias=bias,
    )


@pytest.mark.parametrize("bias", [True, False])
def test_sparse_linear_single_label(bias):
    labels = np.random.randint(low=0, high=DIM, size=BATCH_SIZE, dtype=np.uint32)

    def indices_check(labels, indices):
        assert np.array_equal(indices[:, 0], labels.tensor.numpy())

    run_sparse_linear_test(
        labels=smx.Variable(smx.from_numpy(labels), requires_grad=False),
        indices_check=indices_check,
        exact_param_grad_check=True,
        bias=bias,
    )


@pytest.mark.parametrize("bias", [True, False])
def test_sparse_linear_multi_label(bias):
    label_offsets = np.array([0, 2, 3, 6, 7, 9, 10, 14, 16])
    labels = np.arange(DIM, dtype=np.uint32)
    np.random.shuffle(labels)
    labels = labels[: label_offsets[-1]]
    values = np.ones_like(labels, dtype=np.float32)

    def indices_check(labels, indices):
        offsets = labels.tensor.row_offsets.numpy()
        label_indices = labels.tensor.col_indices.numpy()
        for i in range(len(indices)):
            start = offsets[i]
            end = offsets[i + 1]
            assert np.array_equal(indices[i][: end - start], label_indices[start:end])

    run_sparse_linear_test(
        labels=smx.Variable(
            smx.CsrTensor(label_offsets, labels, values, smx.Shape(BATCH_SIZE, DIM)),
            requires_grad=False,
        ),
        indices_check=indices_check,
        exact_param_grad_check=True,
        bias=bias,
    )


@pytest.mark.parametrize("bias", [True, False])
def test_sparse_linear_disjoint_neurons(bias):
    label_offsets = np.arange(0, BATCH_SIZE * NONZEROS + 1, NONZEROS)
    labels = np.arange(DIM, dtype=np.uint32)
    np.random.shuffle(labels)
    assert len(labels) >= label_offsets[-1]
    labels = labels[: label_offsets[-1]]
    values = np.ones_like(labels, dtype=np.float32)

    def indices_check(labels, indices):
        label_indices = labels.tensor.col_indices.numpy().reshape(BATCH_SIZE, NONZEROS)
        assert np.array_equal(label_indices, indices)

    run_sparse_linear_test(
        labels=smx.Variable(
            smx.CsrTensor(label_offsets, labels, values, smx.Shape(BATCH_SIZE, DIM)),
            requires_grad=False,
        ),
        indices_check=indices_check,
        exact_param_grad_check=True,
        bias=bias,
    )
