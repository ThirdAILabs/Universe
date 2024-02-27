import numpy as np
import pytest
from thirdai import smx

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("bias", [True, False])
def test_smx_embeddings(bias):
    n_embs = 20
    emb_dim = 10

    embs_np = np.random.randint(0, 10000, size=(n_embs, emb_dim)).astype(np.float32)
    bias_np = np.random.randint(0, 10000, size=(emb_dim)).astype(np.float32)

    offsets_np = np.array([0, 3, 7, 9, 13], dtype=np.uint32)
    indices_np = np.array([5, 3, 4, 7, 11, 19, 15, 6, 1, 16, 18, 0, 8], dtype=np.uint32)
    values_np = np.array([1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4])

    indices = smx.CsrTensor(
        row_offsets=smx.from_numpy(offsets_np),
        col_indices=smx.from_numpy(indices_np),
        col_values=smx.from_numpy(values_np),
        dense_shape=smx.Shape(4, 20),
    )

    embs = smx.Variable(smx.from_numpy(embs_np), requires_grad=True)
    bias = smx.Variable(smx.from_numpy(bias_np), requires_grad=True) if bias else None

    out = smx.embedding(smx.Variable(indices, requires_grad=False), embs, bias=bias)

    expected_out = np.stack(
        [
            (1 * embs_np[5] + 2 * embs_np[3] + 3 * embs_np[4]),
            (1 * embs_np[7] + 2 * embs_np[11] + 3 * embs_np[19] + 4 * embs_np[15]),
            (1 * embs_np[6] + 2 * embs_np[1]),
            (1 * embs_np[16] + 2 * embs_np[18] + 3 * embs_np[0] + 4 * embs_np[8]),
        ],
        axis=0,
    )

    if bias:
        expected_out += bias_np

    assert np.allclose(out.tensor.numpy(), expected_out)

    out_grad_np = np.random.randint(0, 10000, size=(4, emb_dim)).astype(np.float32)
    out.backward(smx.from_numpy(out_grad_np))

    expected_grad = np.zeros_like(embs_np)
    expected_grad[5] = 1 * out_grad_np[0]
    expected_grad[3] = 2 * out_grad_np[0]
    expected_grad[4] = 3 * out_grad_np[0]

    expected_grad[7] = 1 * out_grad_np[1]
    expected_grad[11] = 2 * out_grad_np[1]
    expected_grad[19] = 3 * out_grad_np[1]
    expected_grad[15] = 4 * out_grad_np[1]

    expected_grad[6] = 1 * out_grad_np[2]
    expected_grad[1] = 2 * out_grad_np[2]

    expected_grad[16] = 1 * out_grad_np[3]
    expected_grad[18] = 2 * out_grad_np[3]
    expected_grad[0] = 3 * out_grad_np[3]
    expected_grad[8] = 4 * out_grad_np[3]

    assert np.allclose(embs.grad.numpy(), expected_grad)

    if bias:
        assert np.allclose(bias.grad.numpy(), np.sum(out_grad_np, axis=0))
