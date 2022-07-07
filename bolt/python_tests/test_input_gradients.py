from thirdai import bolt
import numpy as np
import pytest

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


@pytest.mark.unit
def test_input_gradients():
    """
    assert that the input gradients are same when required labels are not passed
    and when required labels passed as second best labels ,{1,3,3,3} are second best labels.
    """
    network = build_network()
    set_network_weights_and_biases(network)
    x1 = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]).astype(
        "float32"
    )
    grad1, offset1 = network.get_input_gradients(x1, bolt.CategoricalCrossEntropyLoss())
    grad2, offset2 = network.get_input_gradients(
        x1, bolt.CategoricalCrossEntropyLoss(), np.array([1, 3, 3, 3]).astype("uint32")
    )
    assert grad1 == grad2
