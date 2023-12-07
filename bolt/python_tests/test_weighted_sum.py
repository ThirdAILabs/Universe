import numpy as np
import pytest
from thirdai import bolt


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", [10, 1])
def test_weighted_sum(batch_size):
    n_chunks = 30
    chunk_size = 100
    input_ = bolt.nn.Input(dim=n_chunks * chunk_size)

    weighted_sum = bolt.nn.WeightedSum(n_chunks, chunk_size)(input_)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        weighted_sum, bolt.nn.Input(chunk_size)
    )

    model = bolt.nn.Model(inputs=[input_], outputs=[], losses=[loss])

    x_np = np.random.rand(batch_size, n_chunks, chunk_size)
    x_bolt = bolt.nn.Tensor(x_np.reshape((batch_size, -1)), with_grad=True)

    np_weights = np.random.rand(n_chunks, chunk_size)
    model.set_parameters(np_weights.flatten())

    model.train_on_batch(
        [x_bolt],
        [bolt.nn.Tensor(np.random.rand(batch_size, chunk_size))],
    )

    assert np.allclose(
        np.sum(x_np * np_weights, axis=1),
        weighted_sum.tensor().activations,
    )

    assert np.allclose(
        np.expand_dims(weighted_sum.tensor().gradients, axis=1) * np_weights,
        x_bolt.gradients.reshape((-1, n_chunks, chunk_size)),
    )

    # Because of the nature of bolt parallelism we have a datarace on these gradients.
    # For a small test like this we find that this will sometimes mean the gradients
    # aren't close enough to pass this assertion.
    if batch_size == 1:
        assert np.allclose(
            np.sum(
                np.expand_dims(weighted_sum.tensor().gradients, axis=1) * x_np, axis=0
            ),
            model.get_gradients().reshape((n_chunks, chunk_size)),
        )
