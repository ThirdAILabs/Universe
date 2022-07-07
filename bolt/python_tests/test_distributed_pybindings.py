import pytest

pytestmark = [pytest.mark.integration]

import os
from thirdai import bolt, dataset
import numpy as np
from utils import (
    train_network_distributed,
    build_sparse_hidden_layer_classifier_distributed,
    setup_module,
    load_mnist,
    load_mnist_labels,
)

LEARNING_RATE = 0.0001

# Constructs a bolt network for mnist with a sparse output layer in distributed setting.
def build_sparse_output_layer_network_distributed():
    layers = [
        bolt.FullyConnected(dim=256, activation_function="ReLU"),
        bolt.FullyConnected(
            dim=10,
            sparsity=0.4,
            activation_function="Softmax",
        ),
    ]
    network = bolt.DistributedNetwork(layers=layers, input_dim=784)
    return network


# Constructs a bolt network for mnist with a sparse output layer in distributed setting.
def build_dense_output_layer_network_distributed():
    layers = [
        bolt.FullyConnected(dim=256, activation_function="ReLU"),
        bolt.FullyConnected(
            dim=10,
            activation_function="Softmax",
        ),
    ]
    network = bolt.DistributedNetwork(layers=layers, input_dim=784)
    return network


ACCURACY_THRESHOLD = 0.94
SPARSE_INFERENCE_ACCURACY_THRESHOLD = 0.9
SPARSE_INFERENCE_SPARSE_OUTPUT_ACCURACY_THRESHOLD = 0.35


def test_mnist_sparse_output_layer_distributed():
    network = build_sparse_output_layer_network_distributed()

    train_x, train_y, test_x, test_y = load_mnist()

    train_network_distributed(network, train_x, train_y, epochs=10)

    acc, activations = network.predictSingleNode(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert acc["categorical_accuracy"] >= ACCURACY_THRESHOLD

    # This last check is just to make sure that the accuracy computed in c++ matches
    # what we can compute here using the returned activations. This verifies that the
    # returned activations match and that the metrics are computed correctly.
    predictions = np.argmax(activations, axis=1)

    labels = load_mnist_labels()
    acc_computed = np.mean(predictions == labels)

    assert acc_computed == acc["categorical_accuracy"]


def test_get_set_weights_distributed():

    network = build_sparse_output_layer_network_distributed()
    train_x, train_y, test_x, test_y = load_mnist()

    train_network_distributed(network, train_x, train_y, epochs=10)

    original_acc, _ = network.predictSingleNode(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert original_acc["categorical_accuracy"] >= ACCURACY_THRESHOLD

    untrained_network = build_sparse_output_layer_network_distributed()

    untrained_network.set_weights(0, network.get_weights(0))
    untrained_network.set_weights(1, network.get_weights(1))

    untrained_network.set_biases(0, network.get_biases(0))
    untrained_network.set_biases(1, network.get_biases(1))

    new_acc, _ = untrained_network.predictSingleNode(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert new_acc["categorical_accuracy"] == original_acc["categorical_accuracy"]


def test_get_set_weights_biases_gradients():

    network = build_dense_output_layer_network_distributed()
    train_x, train_y, test_x, test_y = load_mnist()
    learning_rate = 0.0005
    batch_size = network.initTrainSingleNode(
        train_x,
        train_y,
        rehash=3000,
        rebuild=10000,
        verbose=False,
        batch_size=64,
    )

    untrained_network = build_dense_output_layer_network_distributed()

    batch_size = untrained_network.initTrainSingleNode(
        train_x,
        train_y,
        rehash=3000,
        rebuild=10000,
        verbose=False,
        batch_size=64,
    )

    untrained_network.set_weights(0, network.get_weights(0))
    untrained_network.set_biases(0, network.get_biases(0))
    untrained_network.set_weights(1, network.get_weights(1))
    untrained_network.set_biases(1, network.get_biases(1))

    for j in range(batch_size):
        network.calculateGradientSingleNode(j, bolt.CategoricalCrossEntropyLoss())
        untrained_network.set_weights_gradients(0, network.get_weights_gradients(0))
        untrained_network.set_biases_gradients(0, network.get_biases_gradients(0))
        untrained_network.set_weights_gradients(1, network.get_weights_gradients(1))
        untrained_network.set_biases_gradients(1, network.get_biases_gradients(1))
        untrained_network.updateParametersSingleNode(learning_rate)
        network.updateParametersSingleNode(learning_rate)

    old_acc, _ = network.predictSingleNode(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    new_acc, _ = untrained_network.predictSingleNode(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert abs(new_acc["categorical_accuracy"] - old_acc["categorical_accuracy"]) < 0.01
