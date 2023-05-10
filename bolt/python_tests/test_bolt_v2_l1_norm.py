import numpy as np
import pytest
from thirdai import bolt_v2 as bolt


@pytest.mark.unit
def test_l1_normalization():
    shape = (20, 30, 40)

    input_layer = bolt.nn.Input(dims=shape[1:])

    # There is a strange behavior in the fully connected layer code where it modifies
    # it's output gradients during the backward pass with the derivative of the
    # activation function. This is an issue later in the test for checking gradients.
    # Using softmax here solves the issue because we treat its derivative as 1 since
    # its computed in the loss function. This is a hack that should be resolved when
    # the behvaior in bolt is fixed.
    hidden = bolt.nn.FullyConnected(
        dim=shape[-1], input_dim=input_layer.dims()[-1], activation="softmax"
    )(input_layer)

    l1_norm = bolt.nn.L1Normalization()(hidden)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        l1_norm, bolt.nn.Input(l1_norm.dims())
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[], losses=[loss])

    inputs = np.random.randint(0, 10, size=shape).astype(np.float32)
    labels = np.random.randint(0, 10, size=shape).astype(np.float32)

    model.train_on_batch(
        [bolt.nn.Tensor(inputs, with_grad=True)],
        [bolt.nn.Tensor(labels, with_grad=True)],
    )

    assert np.allclose(
        np.ones(shape[:-1]), np.sum(l1_norm.tensor().values, axis=-1)
    )

    l1_norms = np.sum(hidden.tensor().values, axis=-1)
    l1_norms = np.reshape(l1_norms, newshape=(*l1_norms.shape, 1))

    sum_output_grads = np.sum(
        l1_norm.tensor().gradients * l1_norm.tensor().values, axis=-1
    )
    sum_output_grads = np.reshape(
        sum_output_grads, newshape=(*sum_output_grads.shape, 1)
    )

    expected_grads = (l1_norm.tensor().gradients - sum_output_grads) / l1_norms

    assert np.allclose(expected_grads, hidden.tensor().gradients, atol=1e-6)
