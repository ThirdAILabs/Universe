from thirdai import bolt, dataset
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
            sparsity=0.1,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(dim=1, activation_function=bolt.ActivationFunctions.Linear),
    ]

    network = bolt.Network(layers=layers, input_dim=10)

    data = dataset.from_numpy(
        (x_idxs.astype("uint32"), x_vals.astype("float32"), x_offsets.astype("uint32")),
        batch_size=64,
    )
    labels = dataset.from_numpy(
        (y_idxs.astype("uint32"), y_vals.astype("float32"), y_offsets.astype("uint32")),
        batch_size=64,
    )

    batch_size = 256
    learning_rate = 0.001
    epochs = 10
    for i in range(epochs):
        network.train(
            train_data=data,
            train_labels=labels,
            loss_fn=bolt.WeightedMeanAbsolutePercentageError(),
            learning_rate=learning_rate,
            epochs=1,
            verbose=False,
        )
        metrics, _ = network.predict(
            test_data=data,
            test_labels=labels,
            sparse_inference=True,
            metrics=["weighted_mean_absolute_percentage_error"],
            verbose=False,
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
