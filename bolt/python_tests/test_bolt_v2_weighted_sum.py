import numpy as np
import pytest
from thirdai import bolt_v2 as bolt


def create_weighted_sum_model(vectors_shape, weight_shape):
    vectors_input = bolt.nn.Input(dims=vectors_shape)
    weights_input = bolt.nn.Input(dims=weight_shape)

    weighted_sum = bolt.nn.WeightedSum()(vectors_input, weights_input)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        weighted_sum, bolt.nn.Input(weighted_sum.dims())
    )

    model = bolt.nn.Model(
        inputs=[vectors_input, weights_input], outputs=[], losses=[loss]
    )

    return model, weighted_sum


@pytest.mark.unit
def test_single_sum():
    length = 100
    batch_size = 10

    ascending_ints = np.arange(1, length + 1)
    vectors = np.eye(length, dtype=np.float32) * ascending_ints
    weights = (1 / ascending_ints).astype(np.float32)

    vectors_batch = bolt.nn.Tensor(np.tile(vectors, (batch_size, 1, 1)), with_grad=True)
    weights_batch = bolt.nn.Tensor(np.tile(weights, (batch_size, 1)), with_grad=True)
    labels_batch = bolt.nn.Tensor(np.ones((batch_size, length)) * 2)

    model, weighted_sum = create_weighted_sum_model(
        vectors_shape=vectors_batch.dims()[1:], weight_shape=weights_batch.dims()[1:]
    )

    model.train_on_batch([vectors_batch, weights_batch], [labels_batch])

    assert np.allclose(
        np.ones(shape=(batch_size, length), dtype=np.float32),
        weighted_sum.tensor().activations,
    )

    assert np.allclose(
        weighted_sum.tensor().gradients * ascending_ints,
        weights_batch.gradients,
    )

    vector_grads = np.array(
        [np.outer(weights, grad) for grad in weighted_sum.tensor().gradients]
    )

    assert np.allclose(vector_grads, vectors_batch.gradients)
