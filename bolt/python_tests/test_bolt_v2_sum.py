import numpy as np
import pytest
from thirdai import bolt_v2 as bolt

from dataset import create_dataset


@pytest.mark.unit
@pytest.mark.parametrize("sparsity", [0.4, 1.0])
def test_tanh_op(sparsity):
    n_classes = 100
    input_shape = (10, 20, n_classes)

    input_layer = bolt.nn.Input(dims=input_shape[1:])

    hidden_layer = bolt.nn.FullyConnected(
        dim=100, input_dim=n_classes, sparsity=sparsity, activation="linear"
    )(input_layer)

    output_layer = bolt.nn.Sum()(hidden_layer)

    print(output_layer.dims())

    labels = bolt.nn.Input(dim=n_classes)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    train_x, train_y = create_dataset(shape=input_shape, n_batches=2)

    for x, y in zip(train_x, train_y):
        model.train_on_batch(x, y)
        model.update_parameters(0.0001)

        assert output_layer.tensor().indices == None

        if sparsity < 1.0:
            pass
        else:
            assert np.array_equal(
                output_layer.tensor().values(),
                np.sum(hidden_layer.tensor().values, axis=1),
            )
