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


def get_vector(k=-1):
    x1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).astype("float32")
    if k == -1:
        return x1
    return x1[k]


@pytest.mark.unit
def test_input_gradients():
    """
    For a given input and a fixed label on output, the INCREASE in activation of that label,
    when we add a small EPS to the input at each index seperately, should be in the
    same order as the input gradients. For example, let us have an input vector [1,0,0,0] and we choose output label as 1.
    If the input gradients are in the order <2,3,1,0> (the order is on the input indices), then the increase in activation for label 1
    should also be in same order, when we add small EPS at each position seperately.
    """
    network = initialize_network()
    inputs = get_vector()
    gradients, indices = network.get_input_gradients(
        inputs,
        bolt.CategoricalCrossEntropyLoss(),
        required_labels=np.array([1, 1, 1, 1]).astype("uint32"),
    )
    _, act = network.predict(inputs, None)
    """
    For every vector in input,we modify the vector at every position(by adding EPS), and we check the above assertion.
    """
    for input_num in range(len(inputs)):
        modified_vectors = []
        for i in range(len(inputs[input_num])):
            vec = get_vector(input_num)
            vec[i] = vec[i] + 0.001
            modified_vectors.append(vec)
        modified_vectors = np.array(modified_vectors)
        _, vecs_act = network.predict(modified_vectors, None)
        difference = [
            np.array(vec_act) - np.array(act[input_num]) for vec_act in vecs_act
        ]
        activation_difference = [diff[1] for diff in difference]
        assert (
            np.array_equal(
                np.argsort(activation_difference), np.argsort(gradients[input_num])
            )
            == True
        )


@pytest.mark.unit
def test_input_gradients_random_data():
    """
    for "[1,0,0,0]" when these type of vectors given as input and output,
    we expect gradients to be max for non-zero index.
    """
    network = initialize_network()
    inputs, labels = gen_training_data(4, 24000, 0.05)
    times = network.train(
        inputs,
        labels,
        bolt.CategoricalCrossEntropyLoss(),
        learning_rate=0.001,
        epochs=5,
        batch_size=256,
    )
    assert_ratio(network, inputs, labels)
