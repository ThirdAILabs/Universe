import numpy as np
import pytest
from thirdai import bolt


@pytest.mark.unit
def test_quantile_mixing():
    batch_size = 20
    input_dim = 400
    window_size = 50
    frac = 0.2
    input_ = bolt.nn.Input(dim=input_dim)

    quantiles = bolt.nn.QuantileMixing(window_size, frac)(input_)

    loss = bolt.nn.losses.CategoricalCrossEntropy(quantiles, bolt.nn.Input(input_dim))

    model = bolt.nn.Model(inputs=[input_], outputs=[], losses=[loss])

    x_np = np.random.rand(batch_size, input_dim)
    x_bolt = bolt.nn.Tensor(x_np, with_grad=True)

    model.train_on_batch(
        [x_bolt],
        [bolt.nn.Tensor(np.random.rand(batch_size, input_dim))],
    )

    windows = np.reshape(x_np, (batch_size, -1, window_size))
    thresholds = np.sort(windows, axis=-1)[:, :, -int(frac * window_size)]
    mask = np.where(windows >= np.expand_dims(thresholds, -1), 1, 0)
    mask = np.reshape(mask, (batch_size, input_dim))

    assert np.allclose(x_np * mask, quantiles.tensor().activations)

    output_grads = quantiles.tensor().gradients

    assert np.array_equal(output_grads * mask, x_bolt.gradients)
