from thirdai import bolt, dataset
import numpy as np
import pytest
from .utils import (
    assert_activation_difference_and_gradients_in_same_order,
    gen_numpy_training_data,
    get_perturbed_dataset,
    train_network,
)


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
