from thirdai import bolt
import numpy as np
import pytest


def train_bolt_with_wmape(
    x_idxs,
    x_vals,
    x_offsets,
    y_idxs,
    y_vals,
    y_offsets,
):
    layers = [
        bolt.FullyConnected(
            dim=1000,
            load_factor=0.1,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(dim=1, activation_function=bolt.ActivationFunctions.Linear),
    ]

    network = bolt.Network(layers=layers, input_dim=10)

    batch_size = 256
    learning_rate = 0.001
    epochs = 10
    for i in range(epochs):
        network.train(
            x_idxs=x_idxs,
            x_vals=x_vals,
            x_offsets=x_offsets,
            y_idxs=y_idxs,
            y_vals=y_vals,
            y_offsets=y_offsets,
            batch_size=batch_size,
            loss_fn=bolt.WeightedMeanAbsolutePercentageError(),
            learning_rate=learning_rate,
            epochs=1,
            verbose=True,
        )
        metrics, _ = network.predict(
            x_idxs,
            x_vals,
            x_offsets,
            y_idxs,
            y_vals,
            y_offsets,
            batch_size,
            ["weighted_mean_absolute_percentage_error"],
            verbose=True,
        )

    return metrics["weighted_mean_absolute_percentage_error"]


@pytest.mark.unit
def test_wmape_dense_simple():
    n_samples = 10000
    input_dim = 10
    max_out = 100

    labels = np.random.random_integers(max_out, size=(n_samples,)).astype(np.float32)
    examples = np.repeat(labels, input_dim, axis=0)
    examples += np.random.randn(n_samples * input_dim)

    err = train_bolt_with_wmape(
        x_idxs=np.concatenate([np.arange(10) for _ in range(10000)]),
        x_vals=examples,
        x_offsets=np.arange(0, n_samples * input_dim + 1, input_dim),
        y_idxs=np.zeros(shape=(n_samples,)),
        y_vals=labels,
        y_offsets=np.arange(0, n_samples + 1, 1),
    )

    assert err < 0.1


@pytest.mark.unit
def test_wmape_one_hot_simple():
    n_samples = 10000
    input_dim = 10
    max_out = input_dim - 1

    labels = np.random.random_integers(max_out, size=(n_samples,))
    y_vals = labels.astype(np.float32) + 0.1 * np.random.randn(n_samples)

    err = train_bolt_with_wmape(
        x_idxs=labels,
        x_vals=np.ones(shape=(n_samples,)),
        x_offsets=np.arange(0, n_samples + 1, 1),
        y_idxs=np.zeros(shape=(n_samples,)),
        y_vals=y_vals,
        y_offsets=np.arange(0, n_samples + 1, 1),
    )
    print(err)
    assert err < 0.1


# test_wmape_dense_simple()
# test_wmape_one_hot_simple()
