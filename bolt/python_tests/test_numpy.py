# Add unit and release test marker for all tests in this file
from thirdai import bolt
import numpy as np
import pytest

from utils import gen_training_data

pytestmark = [pytest.mark.unit, pytest.mark.release]


def train_simple_bolt_model(examples, labels, sparsity=1, n_classes=10):
    layers = [
        bolt.FullyConnected(
            dim=n_classes,
            sparsity=sparsity,
            activation_function=bolt.ActivationFunctions.Softmax,
        )
    ]
    network = bolt.Network(layers=layers, input_dim=n_classes)

    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    network.train(
        train_data=examples,
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

    # Check that predict functions correctly and returns activations when
    # no labels are specified.
    _, activations = network.predict(
        examples, None, batch_size, ["categorical_accuracy"], verbose=False
    )
    preds = np.argmax(activations, axis=1)
    acc_computed = np.mean(preds == labels)

    assert acc_computed == acc["categorical_accuracy"]

    return acc["categorical_accuracy"]


def train_simple_bolt_model_non_trainable_hidden_layer(
    examples, labels, load_factor=1, n_classes=10
):
    layers = [
        bolt.FullyConnected(
            dim=100,
            sparsity=load_factor,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(
            dim=n_classes,
            sparsity=load_factor,
            activation_function=bolt.ActivationFunctions.Softmax,
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=n_classes)

    batch_size = 64
    learning_rate = 0.001
    epochs = 100

    before_training_weigths = network.get_weights(0)
    network.setTrainable(layer_index=0, trainable=False)

    network.train(
        train_data=examples,
        train_labels=labels,
        batch_size=batch_size,
        loss_fn=bolt.CategoricalCrossEntropyLoss(),
        learning_rate=learning_rate,
        epochs=epochs,
        verbose=False,
    )
    after_training_weigths = network.get_weights(0)

    acc, _ = network.predict(
        examples, labels, batch_size, ["categorical_accuracy"], verbose=False
    )

    return acc["categorical_accuracy"], np.linalg.norm(
        after_training_weigths - before_training_weigths
    )


def train_sparse_bolt_model(
    x_idxs, x_vals, x_offsets, y_idxs, y_vals, y_offsets, inp_dim, n_classes
):
    layers = [
        bolt.FullyConnected(
            dim=n_classes,
            sparsity=1,
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
        train_data=(x_idxs, x_vals, x_offsets),
        train_labels=(y_idxs, y_vals, y_offsets),
        batch_size=batch_size,
        loss_fn=bolt.CategoricalCrossEntropyLoss(),
        learning_rate=learning_rate,
        epochs=epochs,
        verbose=False,
    )
    acc, _ = network.predict(
        test_data=(x_idxs, x_vals, x_offsets),
        test_labels=(y_idxs, y_vals, y_offsets),
        batch_size=batch_size,
        metrics=["categorical_accuracy"],
        verbose=False,
    )
    ##
    return acc["categorical_accuracy"]


@pytest.mark.unit
def test_read_easy_mock_data():
    """
    Generates easy mock dataset as a numpy array and asserts that BOLT performs well.
    """
    labels, examples, n_classes = gen_training_data(n_classes=10, n_samples=1000)
    acc = train_simple_bolt_model(examples, labels)
    assert acc > 0.8


@pytest.mark.unit
def test_mock_data_non_trainable_hidden_layer():
    """
    Generates easy mock dataset as a numpy array and asserts that BOLT performs well.
    also asserts that the weights of the non-trainable layer have not changed
    """
    n_classes = 10
    n_samples = 1000
    possible_one_hot_encodings = np.eye(n_classes)
    labels = np.random.choice(n_classes, size=n_samples)
    examples = possible_one_hot_encodings[labels]
    noise = np.random.normal(0, 0.1, examples.shape)
    examples = examples + noise

    acc, norm = train_simple_bolt_model_non_trainable_hidden_layer(examples, labels)
    assert acc > 0.8
    assert norm == 0.0


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
    assert acc > 0.8


@pytest.mark.unit
def test_easy_sparse_layer():
    """
    Generates easy mock dataset as a numpy array and asserts that BOLT performs well trained with a sparse output.
    """
    labels, examples, n_classes = gen_training_data(n_classes=100, n_samples=10000)
    acc = train_simple_bolt_model(examples, labels, sparsity=0.1, n_classes=n_classes)
    assert acc > 0.8


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
