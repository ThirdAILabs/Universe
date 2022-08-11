from thirdai import bolt, dataset
import numpy as np
import pytest
from .utils import (
    assert_activation_difference_and_gradients_in_same_order,
    gen_numpy_training_data,
    gen_random_weights_simple_network,
    gen_random_bias_simple_network,
    get_perturbed_dataset,
    train_network,
)

np.random.seed(17)


def build_network():
    layers = [
        bolt.FullyConnected(
            dim=3,
            activation_function="relu",
        ),
        bolt.FullyConnected(
            dim=4,
            activation_function="softmax",
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=4)
    return network


def set_network_weights_and_biases(network):
    w1, w2 = gen_random_weights_simple_network(
        input_output_layer_dim=4, hidden_layer_dim=3
    )
    b1, b2 = gen_random_bias_simple_network(output_layer_dim=4, hidden_layer_dim=3)
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
    numpy_inputs, numpy_labels = gen_numpy_training_data(
        4, convert_to_bolt_dataset=False
    )
    inputs = dataset.from_numpy(numpy_inputs, batch_size=256)
    labels = dataset.from_numpy(numpy_labels, batch_size=256)
    train_network(network, inputs, labels, epochs=5)
    gradients = network.get_input_gradients(
        inputs,
        bolt.CategoricalCrossEntropyLoss(),
        required_labels=numpy_labels,
    )
    _, act = network.predict(inputs, None)
    """
    For every vector in input,we modify the vector at every position(by adding EPS), and we check the above assertion.
    """
    for input_num in range(len(numpy_inputs)):
        perturbed_dataset = get_perturbed_dataset(numpy_inputs[input_num])
        _, perturbed_activations = network.predict(perturbed_dataset, None)
        assert_activation_difference_and_gradients_in_same_order(
            perturbed_activations,
            numpy_labels[input_num],
            gradients[input_num],
            act[input_num],
        )


@pytest.mark.unit
def test_return_indices_for_sparse_and_dense_inputs():
    """
    For Dense inputs we should not return indices but for sparse inputs we should return sparse indices.
    """
    dense_numpy_inputs = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).astype("float32")
    sparse_numpy_inputs = (
        np.array([0, 0, 1, 3, 0, 1, 0, 1, 2, 3]).astype("uint32"),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype("float32"),
        np.array([0, 1, 4, 6, 10]).astype("uint32"),
    )
    dense_inputs = dataset.from_numpy(dense_numpy_inputs, batch_size=4)
    sparse_inputs = dataset.from_numpy(sparse_numpy_inputs, batch_size=4)
    network = initialize_network()
    dense_inputs_gradients = network.get_input_gradients(
        dense_inputs, bolt.CategoricalCrossEntropyLoss()
    )
    assert len(dense_inputs_gradients) == len(dense_numpy_inputs)
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

    assert combined_sparse_indices.all() == (sparse_numpy_inputs[0]).all()
