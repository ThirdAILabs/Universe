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


def initialize_network():
    network = build_network()
    set_network_weights_and_biases(network)
    return network


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
    inputs = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).astype("float32")
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
            """
            We are making a copy because in python assign operation makes two variables to point
            to same address space, and we only want to modify one and keep the other same.
            """
            vec = np.array(inputs[input_num])
            vec[i] = vec[i] + 0.001
            modified_vectors.append(vec)
        modified_vectors = np.array(modified_vectors)
        _, vecs_act = network.predict(modified_vectors, None)
        act_difference_at_label_one = [
            np.array(vec_act[1]) - np.array(act[input_num][1]) for vec_act in vecs_act
        ]
        assert np.array_equal(
            np.argsort(act_difference_at_label_one),
            np.argsort(gradients[input_num]),
        )


@pytest.mark.unit
def test_return_indices_for_sparse_and_dense_inputs():
    """
    For Dense inputs we should not return indices but for sparse inputs we should return sparse indices.
    """
    dense_inputs = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).astype("float32")
    sparse_inputs = (
        np.array([0, 0, 1, 3, 0, 1, 0, 1, 2, 3]).astype("uint32"),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype("float32"),
        np.array([0, 1, 4, 6, 10]).astype("uint32"),
    )
    network = initialize_network()
    _, dense_inputs_indices = network.get_input_gradients(
        dense_inputs, bolt.CategoricalCrossEntropyLoss()
    )
    assert not (dense_inputs_indices)
    _, sparse_inputs_indices = network.get_input_gradients(
        sparse_inputs, bolt.CategoricalCrossEntropyLoss()
    )
    combined_sparse_indices = np.array(
        [
            index
            for sparse_input_indices in sparse_inputs_indices
            for index in sparse_input_indices
        ]
    )

    assert combined_sparse_indices.all() == (sparse_inputs[0]).all()
