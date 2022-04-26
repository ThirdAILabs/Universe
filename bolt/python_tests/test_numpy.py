import numpy as np
import pytest
from thirdai import bolt


def train_simple_bolt_model(examples, labels):
    layers = [
        bolt.FullyConnected(
            dim=10, load_factor=1, activation_function=bolt.ActivationFunctions.Softmax
        )
    ]
    network = bolt.Network(layers=layers, input_dim=10)

    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    network.train(
        train_examples=examples,
        train_labels=labels,
        batch_size=batch_size,
        loss_fn=bolt.CategoricalCrossEntropyLoss(),
        learning_rate=learning_rate,
        epochs=epochs,
        verbose=False,
    )
    acc, _ = network.predict(
        examples, labels, batch_size, ["categorical_accuracy"], verbose=False
    )

    return acc["categorical_accuracy"]


def train_sparse_bolt_model(
    x_idxs, x_vals, x_offsets, y_idxs, y_vals, y_offsets, inp_dim, n_classes
):
    layers = [
        bolt.FullyConnected(
            dim=n_classes,
            load_factor=1,
            activation_function=bolt.getActivationFunction("ReLU"),
        )
    ]
    network = bolt.Network(layers=layers, input_dim=inp_dim)
    ##
    batch_size = 64
    learning_rate = 0.01
    epochs = 5
    ##
    network.train(
        x_idxs=x_idxs,
        x_vals=x_vals,
        x_offsets=x_offsets,
        y_idxs=y_idxs,
        y_vals=y_vals,
        y_offsets=y_offsets,
        batch_size=batch_size,
        loss_fn=bolt.CategoricalCrossEntropyLoss(),
        learning_rate=learning_rate,
        epochs=epochs,
        verbose=False,
    )
    acc, _ = network.predict(
        x_idxs,
        x_vals,
        x_offsets,
        y_idxs,
        y_vals,
        y_offsets,
        batch_size,
        ["categorical_accuracy"],
        verbose=False,
    )
    ##
    return acc["categorical_accuracy"]


@pytest.mark.unit
def test_read_easy_mock_data():
    """
    Generates easy mock dataset as a numpy array and asserts that BOLT performs well.
    """
    n_classes = 10
    n_samples = 1000
    possible_one_hot_encodings = np.eye(n_classes)
    labels = np.random.choice(n_classes, size=n_samples)
    examples = possible_one_hot_encodings[labels]
    noise = np.random.normal(0, 0.1, examples.shape)
    examples = examples + noise

    acc = train_simple_bolt_model(examples, labels)
    assert acc > 0.99


@pytest.mark.unit
def test_mock_sparse_data():
    """
    Generates easy mock dataset for a sparse input to BOLT.
    """
    inp_dim = 100
    n_classes = 10
    n_samples = 50
    x_idxs = np.arange(2 * n_samples)
    np.random.shuffle(x_idxs)
    x_idxs %= inp_dim
    x_vals = np.ones(2 * n_samples) + 0.1 * np.random.rand(2 * n_samples)
    x_offsets = 2 * np.arange(n_samples + 1)
    y_idxs = np.arange(2 * n_samples)
    np.random.shuffle(y_idxs)
    y_idxs %= n_classes
    y_vals = np.ones(2 * n_samples) + 0.1 * np.random.rand(2 * n_samples)
    y_offsets = 2 * np.arange(n_samples + 1)
    acc = train_sparse_bolt_model(
        x_idxs, x_vals, x_offsets, y_idxs, y_vals, y_offsets, inp_dim, n_classes
    )
    assert acc > 0.9


@pytest.mark.unit
def test_read_noise():
    """
    Generates random noise as a numpy array and asserts that BOLT cannot perform well.
    """
    n_classes = 10
    n_samples = 1000
    labels = np.random.choice(n_classes, size=n_samples)
    examples = np.random.normal(0, 1, (n_samples, n_classes))

    acc = train_simple_bolt_model(examples, labels)
    assert acc < 0.2
