import numpy as np
import pytest
from thirdai import bolt_v2 as bolt


def create_weighted_sum_model(emb_shape, weights_shape):
    emb_input = bolt.nn.Input(dims=emb_shape)
    weights_input = bolt.nn.Input(dims=weights_shape)

    weighted_sum = bolt.nn.WeightedSum()(emb_input, weights_input)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        weighted_sum, bolt.nn.Input(weighted_sum.dims())
    )

    model = bolt.nn.Model(inputs=[emb_input, weights_input], outputs=[], losses=[loss])

    return model, weighted_sum


@pytest.mark.unit
def test_single_weighted_sum():
    length = 100
    batch_size = 10

    ascending_ints = np.arange(1, length + 1)
    embs = np.eye(length, dtype=np.float32) * ascending_ints
    weights = (1 / ascending_ints).astype(np.float32)

    emb_batch = bolt.nn.Tensor(np.tile(embs, (batch_size, 1, 1)), with_grad=True)
    weights_batch = bolt.nn.Tensor(np.tile(weights, (batch_size, 1)), with_grad=True)
    labels_batch = bolt.nn.Tensor(np.ones((batch_size, length)) * 2)

    model, weighted_sum = create_weighted_sum_model(
        emb_shape=emb_batch.dims()[1:], weights_shape=weights_batch.dims()[1:]
    )

    model.train_on_batch([emb_batch, weights_batch], [labels_batch])

    assert np.allclose(
        np.ones(shape=(batch_size, length), dtype=np.float32),
        weighted_sum.tensor().activations,
    )

    assert np.allclose(
        weighted_sum.tensor().gradients * ascending_ints,
        weights_batch.gradients,
    )

    emb_grads = np.array(
        [np.outer(weights, grad) for grad in weighted_sum.tensor().gradients]
    )

    assert np.allclose(emb_grads, emb_batch.gradients)


@pytest.mark.unit
def test_nd_weighted_sum():
    batch_size = 10
    seq_len = 20
    emb_dim = 50

    emb_shape = (batch_size, seq_len, emb_dim)
    weights_shape = (batch_size, 5, 6, seq_len)
    labels_shape = (*weights_shape[:-1], emb_dim)

    model, weighted_sum = create_weighted_sum_model(
        emb_shape=emb_shape[1:], weights_shape=weights_shape[1:]
    )

    embs = np.random.randint(0, 10, size=emb_shape).astype(np.float32)
    weights = np.random.randint(0, 10, size=weights_shape).astype(np.float32)
    labels = np.random.randint(0, 10, size=labels_shape).astype(np.float32)

    emb_batch = bolt.nn.Tensor(embs, with_grad=True)
    weights_batch = bolt.nn.Tensor(weights, with_grad=True)
    labels_batch = bolt.nn.Tensor(labels)

    model.train_on_batch([emb_batch, weights_batch], [labels_batch])

    expected_output = np.einsum("bijn,bnd -> bijd", weights, embs)

    assert expected_output.shape == labels_shape

    assert np.allclose(expected_output, weighted_sum.tensor().activations)

    expected_weights_grad = np.einsum(
        "bijd,bnd -> bijn", weighted_sum.tensor().gradients, emb_batch.activations
    )

    assert np.allclose(expected_weights_grad, weights_batch.gradients)

    expected_emb_grad = np.einsum(
        "bijd,bijn -> bnd", weighted_sum.tensor().gradients, weights_batch.activations
    )

    assert np.allclose(expected_emb_grad, emb_batch.gradients)
