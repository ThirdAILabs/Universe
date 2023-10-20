import numpy as np
import pytest
from thirdai import bolt


@pytest.mark.unit
def test_sparsify():
    BATCH_SIZE = 4
    DIM = 8
    SPARSITY = 0.25

    input_layer = bolt.nn.Input(DIM)

    sparsify = bolt.nn.Sparsify(SPARSITY)(input_layer)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        sparsify, bolt.nn.Input(sparsify.dim())
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[], losses=[loss])

    inputs = np.random.rand(BATCH_SIZE, DIM).astype(np.float32)
    labels = np.random.rand(BATCH_SIZE, DIM).astype(np.float32)

    model.train_on_batch(
        [bolt.nn.Tensor(inputs, with_grad=True)], [bolt.nn.Tensor(labels)]
    )

    output_indices = sparsify.tensor().active_neurons
    output_values = sparsify.tensor().activations

    expected_indices = np.sort(np.argsort(inputs)[:, -int(DIM * SPARSITY) :])
    expected_values = inputs[
        np.arange(len(expected_indices))[:, None], expected_indices
    ]

    assert np.array_equal(output_indices, expected_indices)
    assert np.allclose(output_values, expected_values)

    expected_gradients = np.zeros_like(inputs)
    expected_gradients[
        np.arange(len(expected_indices))[:, None], expected_indices
    ] = sparsify.tensor().gradients

    assert np.allclose(input_layer.tensor().gradients, expected_gradients)
