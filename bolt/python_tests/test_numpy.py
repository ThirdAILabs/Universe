import numpy as np
import pytest
from thirdai import bolt


def train_simple_bolt_model(examples, labels):
    layers = [
        bolt.LayerConfig(
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

    return acc["categorical_accuracy"][0]


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
