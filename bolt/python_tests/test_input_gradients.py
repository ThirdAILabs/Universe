from thirdai import bolt
import numpy as np
import pytest
from .utils import gen_training_data

pytestmark = [pytest.mark.unit, pytest.mark.release]


def build_network():
    layers = [
        bolt.FullyConnected(
            dim=3,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(
            dim=4,
            activation_function=bolt.ActivationFunctions.Softmax,
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=4)
    return network


def gen_random_weights():
    np.random.seed(17)
    w1 = np.random.randn(3, 4).astype(np.float32)
    w2 = np.random.randn(4, 3).astype(np.float32)
    return w1, w2


def gen_random_bias():
    b1 = np.random.randn(3).astype(np.float32)
    b2 = np.random.randn(4).astype(np.float32)
    return b1, b2


def set_network_weights_and_biases(network):
    w1, w2 = gen_random_weights()
    b1, b2 = gen_random_bias()
    network.set_weights(0, w1)
    network.set_biases(0, b1)
    network.set_weights(1, w2)
    network.set_biases(1, b2)


def initialize_network():
    network = build_network()
    set_network_weights_and_biases(network)
    return network


def assert_ratio(network, x1, labels):
    """
    checking the gradients are highest for the label mentioned in labels array
    for most of the cases.
    """
    gradients, indices = network.get_input_gradients(
        x1,
        bolt.CategoricalCrossEntropyLoss(),
        required_labels=labels,
    )
    max_times = 0
    total = len(gradients)
    for i in range(total):
        abs_list = list(map(abs, gradients[i]))
        index = abs_list.index(max(abs_list))
        if index == labels[i]:
            max_times += 1
    assert (max_times / total) > 0.7


@pytest.mark.unit
def test_input_gradients_sample_data():
    """
    for "[1,0,0,0]" these type of inputs, we expect gradients
    to be max for non-zero index, and for [1,1,0,0] the label mentioned '1' but for
    [1,0,0,0] label mentioned '0' so the second index element has most influence over
    the input to flip the label, so expected to have high gradient.
    """
    network = initialize_network()
    x1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]
    ).astype("float32")
    labels = np.array([0, 1, 2, 3, 1, 2, 1]).astype("uint32")
    assert_ratio(network, x1, labels)


@pytest.mark.unit
def test_input_gradients_random_data():
    """
    we check the same thing as above but for bigger and random dataset.
    """
    network = initialize_network()
    x1, labels = gen_training_data(4, 24000, 0.05)
    times = network.train(
        x1,
        labels,
        bolt.CategoricalCrossEntropyLoss(),
        learning_rate=0.001,
        epochs=5,
        batch_size=256,
    )
    assert_ratio(network, x1, labels)
