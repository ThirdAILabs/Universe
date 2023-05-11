import numpy as np
import pytest
from thirdai import bolt_v2 as bolt

from dataset import create_dataset


@pytest.mark.unit
@pytest.mark.parametrize("sparsity", [0.4, 1.0])
def test_tanh_op(sparsity):
    n_classes = 100
    input_shape = (10, 20, 30, n_classes)

    input_layer = bolt.nn.Input(dims=input_shape[1:])

    hidden_layer = bolt.nn.FullyConnected(
        dim=100, input_dim=n_classes, sparsity=sparsity, activation="linear"
    )(input_layer)

    output_layer = bolt.nn.Tanh()(hidden_layer)

    labels = bolt.nn.Input(dims=input_shape[1:])

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    train_x, train_y = create_dataset(shape=input_shape, n_batches=2)

    for x, y in zip(train_x, train_y):
        model.train_on_batch(x, y)
        model.update_parameters(0.0001)

        if sparsity < 1.0:
            assert np.array_equal(
                hidden_layer.tensor().indices, output_layer.tensor().indices
            )
        else:
            assert hidden_layer.tensor().indices == None
            assert output_layer.tensor().indices == None

        assert np.allclose(
            np.tanh(hidden_layer.tensor().values), output_layer.tensor().values
        )

        # d/dx tanh(x) = 1 - tanh^2(x)
        expected_grads = output_layer.tensor().gradients * (
            1 - np.square(output_layer.tensor().values)
        )

        assert np.allclose(hidden_layer.tensor().gradients, expected_grads)
