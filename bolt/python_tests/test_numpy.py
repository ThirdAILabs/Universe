# Add unit and release test marker for all tests in this file
from thirdai import bolt, dataset
import numpy as np
import pytest

from .utils import gen_numpy_training_data

pytestmark = [pytest.mark.unit, pytest.mark.release]


def train_simple_bolt_model(
    examples_np, labels_np, sparsity=1, n_classes=10, batch_size=64
):

    examples = dataset.from_numpy(examples_np, batch_size)
    labels = dataset.from_numpy(labels_np, batch_size)

    layers = [
        bolt.FullyConnected(
            dim=n_classes,
            sparsity=sparsity,
            activation_function=bolt.ActivationFunctions.Softmax,
        )
    ]
    network = bolt.Network(layers=layers, input_dim=n_classes)

    learning_rate = 0.001
    epochs = 5

    network.train(
        train_data=examples,
        train_labels=labels,
        loss_fn=bolt.CategoricalCrossEntropyLoss(),
        learning_rate=learning_rate,
        epochs=epochs,
        verbose=False,
    )
    acc, _ = network.predict(
        examples, labels, metrics=["categorical_accuracy"], verbose=False
    )
    # Check that predict functions correctly and returns activations when
    # no labels are specified.
    _, activations = network.predict(
        examples, None, metrics=["categorical_accuracy"], verbose=False
    )
    preds = np.argmax(activations, axis=1)
    acc_computed = np.mean(preds == labels_np)

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

    learning_rate = 0.001
    epochs = 100

    before_training_weigths = network.get_weights(0)
    network.setTrainable(layer_index=0, trainable=False)

    network.train(
        train_data=examples,
        train_labels=labels,
        loss_fn=bolt.CategoricalCrossEntropyLoss(),
        learning_rate=learning_rate,
        epochs=epochs,
        verbose=False,
    )
    after_training_weigths = network.get_weights(0)

    acc, _ = network.predict(
        examples, labels, metrics=["categorical_accuracy"], verbose=False
    )

    return acc["categorical_accuracy"], np.linalg.norm(
        after_training_weigths - before_training_weigths
    )


def train_sparse_bolt_model(
    x_idxs, x_vals, x_offsets, y_idxs, y_vals, y_offsets, inp_dim, n_classes
):
    data = dataset.from_numpy((x_idxs, x_vals, x_offsets), batch_size=64)
    labels = dataset.from_numpy((y_idxs, y_vals, y_offsets), batch_size=64)
    layers = [
        bolt.FullyConnected(
            dim=n_classes,
            sparsity=1,
            activation_function=bolt.getActivationFunction("ReLU"),
        )
    ]
    network = bolt.Network(layers=layers, input_dim=inp_dim)
    ##
    learning_rate = 0.01
    epochs = 5
    ##
    network.train(
        train_data=data,
        train_labels=labels,
        loss_fn=bolt.CategoricalCrossEntropyLoss(),
        learning_rate=learning_rate,
        epochs=epochs,
        verbose=False,
    )
    acc, _ = network.predict(
        test_data=data,
        test_labels=labels,
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
    examples_np, labels_np = gen_numpy_training_data(
        n_classes=10, n_samples=1000, convert_to_bolt_dataset=False
    )
    acc = train_simple_bolt_model(examples_np=examples_np, labels_np=labels_np)
    assert acc > 0.8


@pytest.mark.unit
def test_mock_data_non_trainable_hidden_layer():
    """
    Generates easy mock dataset as a numpy array and asserts that BOLT performs well.
    also asserts that the weights of the non-trainable layer have not changed
    """
    examples, labels = gen_numpy_training_data(n_classes=10, n_samples=10000)
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
        x_idxs.astype("uint32"),
        x_vals.astype("float32"),
        x_offsets.astype("uint32"),
        y_idxs.astype("uint32"),
        y_vals.astype("float32"),
        y_offsets.astype("uint32"),
        inp_dim,
        n_classes,
    )
    assert acc > 0.8


@pytest.mark.unit
def test_easy_sparse_layer():
    """
    Generates easy mock dataset as a numpy array and asserts that BOLT performs well trained with a sparse output.
    """
    examples_np, labels_np = gen_numpy_training_data(
        n_classes=100, n_samples=10000, convert_to_bolt_dataset=False
    )
    acc = train_simple_bolt_model(
        examples_np=examples_np, labels_np=labels_np, sparsity=0.1, n_classes=100
    )
    assert acc > 0.8


@pytest.mark.unit
def test_read_noise():
    """
    Generates random noise as a numpy array and asserts that BOLT cannot perform well.
    """
    n_classes = 10
    n_samples = 1000
    labels_np = np.random.choice(n_classes, size=n_samples).astype("uint32")
    examples_np = np.random.normal(0, 1, (n_samples, n_classes)).astype("float32")

    acc = train_simple_bolt_model(examples_np=examples_np, labels_np=labels_np)
    assert acc < 0.2
