import numpy as np
import pytest
from thirdai import bolt_v2 as bolt

from dataset import create_dataset


def concat_active_neurons(computations):
    arrays = []
    total_dim = 0
    for comp in computations:
        if comp.tensor().active_neurons is None:
            arr = np.tile(
                np.arange(
                    start=total_dim, stop=total_dim + comp.dims()[-1], dtype=np.uint32
                ),
                (*comp.tensor().dims()[:-1], 1),
            )
            arrays.append(arr)
        else:
            arrays.append(comp.tensor().active_neurons + total_dim)

        total_dim += comp.dims()[-1]

    return np.concatenate(arrays, axis=-1, dtype=np.uint32)


def concat_activations(computations):
    return np.concatenate(
        [comp.tensor().activations for comp in computations], axis=-1, dtype=np.float32
    )


def concat_gradients(computations):
    return np.concatenate(
        [comp.tensor().gradients for comp in computations], axis=-1, dtype=np.float32
    )


@pytest.mark.unit
@pytest.mark.parametrize("use_sparsity", [True, False])
def test_concatenation_op(use_sparsity):
    n_classes = 100
    input_shape = (10, 20, 30, n_classes)

    input_layer = bolt.nn.Input(dims=input_shape[1:])

    hidden_layers = []
    for dim, sparsity in [(10, 1.0), (20, 0.4), (10, 0.5), (5, 1.0)]:
        if not use_sparsity:
            sparsity = 1.0
        layer = bolt.nn.FullyConnected(
            dim=dim, input_dim=n_classes, sparsity=sparsity, activation="linear"
        )
        hidden_layers.append(layer(input_layer))

    concat = bolt.nn.Concatenate()(hidden_layers)

    output = bolt.nn.FullyConnected(dim=n_classes, input_dim=45, activation="softmax")(
        concat
    )

    labels = bolt.nn.Input(dims=input_shape[1:])

    loss = bolt.nn.losses.CategoricalCrossEntropy(activations=output, labels=labels)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    train_x, train_y = create_dataset(shape=input_shape, n_batches=10)

    for x, y in zip(train_x, train_y):
        model.train_on_batch(x, y)
        model.update_parameters(0.0001)

        if use_sparsity:
            assert np.array_equal(
                concat_active_neurons(hidden_layers), concat.tensor().active_neurons
            )
        else:
            assert concat.tensor().active_neurons == None

        assert np.array_equal(
            concat_activations(hidden_layers), concat.tensor().activations
        )

        assert np.array_equal(
            concat_gradients(hidden_layers), concat.tensor().gradients
        )
