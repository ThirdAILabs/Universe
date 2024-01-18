import numpy as np
import pytest
from thirdai import bolt


@pytest.mark.unit
def test_max_pool_1d():
    batch_size = 20
    input_dim = 400
    window_size = 8
    input_ = bolt.nn.Input(dim=input_dim)

    max_pool = bolt.nn.MaxPool1D(window_size)(input_)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        max_pool, bolt.nn.Input(input_dim // window_size)
    )

    model = bolt.nn.Model(inputs=[input_], outputs=[], losses=[loss])

    x_np = np.random.rand(batch_size, input_dim)
    x_bolt = bolt.nn.Tensor(x_np, with_grad=True)

    model.train_on_batch(
        [x_bolt],
        [bolt.nn.Tensor(np.random.rand(batch_size, input_dim // window_size))],
    )

    expected_output = np.max(
        x_np.reshape(batch_size, -1, window_size), axis=-1
    ).reshape(batch_size, -1)

    assert np.allclose(expected_output, max_pool.tensor().activations)

    max_indices = np.argmax(
        x_np.reshape(batch_size, -1, window_size), axis=-1, keepdims=True
    )
    output_grads = np.expand_dims(max_pool.tensor().gradients, axis=2)

    input_grads = np.zeros((batch_size, input_dim // window_size, window_size))

    input_grads[
        np.arange(batch_size)[:, None, None],
        np.arange(input_dim // window_size)[None, :, None],
        max_indices,
    ] = output_grads

    assert np.array_equal(x_bolt.gradients, input_grads.reshape(batch_size, -1))
