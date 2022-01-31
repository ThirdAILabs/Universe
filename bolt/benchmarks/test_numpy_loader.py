import numpy as np
import pytest
from thirdai import bolt

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

    layers = [
        bolt.LayerConfig(dim=10, load_factor=1,
                            activation_function="Softmax")
    ]
    network = bolt.Network(layers=layers, input_dim=10)

    network.train(examples, labels, 64, 0.001, 5)
    acc = network.predict(examples, labels, 64)
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

    layers = [
        bolt.LayerConfig(dim=10, load_factor=1,
                            activation_function="Softmax")
    ]
    network = bolt.Network(layers=layers, input_dim=10)

    network.train(examples, labels, 64, 0.001, 5)
    acc = network.predict(examples, labels, 64)
    assert acc < 0.2


