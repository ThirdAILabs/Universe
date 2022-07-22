import pytest


pytestmark = [pytest.mark.unit, pytest.mark.release]

from thirdai import bolt, dataset
import numpy as np


from .utils import (
    train_single_node_distributed_network,
    copy_two_layer_network_parameters,
    gen_training_data,
)

ACCURACY_THRESHOLD = 0.8


def build_simple_bolt_network(sparsity=1, n_classes=10):
    layers = [
        bolt.FullyConnected(
            dim=50,
            sparsity=1,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(
            dim=n_classes,
            sparsity=sparsity,
            activation_function=bolt.ActivationFunctions.Softmax,
        ),
    ]
    network = bolt.DistributedNetwork(layers=layers, input_dim=n_classes)
    return network


def train_multiple_networks_same_gradients(
    network, untrained_network, num_of_batches, test_x, test_y
):
    learning_rate = 0.0005
    for batch_num in range(num_of_batches):
        network.calculateGradientSingleNode(
            batch_idx=batch_num, loss_fn=bolt.CategoricalCrossEntropyLoss()
        )
        untrained_network.set_weights_gradients(
            layer_index=0,
            new_weights_gradients=network.get_weights_gradients(layer_index=0),
        )
        untrained_network.set_biases_gradients(
            layer_index=0,
            new_biases_gradients=network.get_biases_gradients(layer_index=0),
        )
        untrained_network.set_weights_gradients(
            layer_index=1,
            new_weights_gradients=network.get_weights_gradients(layer_index=1),
        )
        untrained_network.set_biases_gradients(
            layer_index=1,
            new_biases_gradients=network.get_biases_gradients(layer_index=1),
        )
        untrained_network.updateParametersSingleNode(learning_rate)
        network.updateParametersSingleNode(learning_rate)

    old_acc, _ = network.predictSingleNode(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    new_acc, _ = untrained_network.predictSingleNode(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert (
        new_acc["categorical_accuracy"] > ACCURACY_THRESHOLD
        and new_acc["categorical_accuracy"] == old_acc["categorical_accuracy"]
    )


def test_simple_bolt_network_distributed():
    network = build_simple_bolt_network()

    train_x, train_y = gen_training_data()
    test_x, test_y = gen_training_data(n_samples=100)

    train_single_node_distributed_network(network, train_x, train_y, epochs=10)

    acc, activations = network.predictSingleNode(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )
    assert acc["categorical_accuracy"] >= ACCURACY_THRESHOLD


def test_get_set_weights_distributed():

    network = build_simple_bolt_network()
    train_x, train_y = gen_training_data()
    test_x, test_y = gen_training_data(n_samples=100)

    train_single_node_distributed_network(network, train_x, train_y, epochs=10)

    original_acc, _ = network.predictSingleNode(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )
    assert original_acc["categorical_accuracy"] >= ACCURACY_THRESHOLD

    untrained_network = build_simple_bolt_network()

    copy_two_layer_network_parameters(network, untrained_network)

    new_acc, _ = untrained_network.predictSingleNode(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )
    assert new_acc["categorical_accuracy"] == original_acc["categorical_accuracy"]


def test_basic_gradient_sharing():

    network = build_simple_bolt_network()

    train_x, train_y = gen_training_data()
    test_x, test_y = gen_training_data(n_samples=100)

    num_of_batches = network.prepareNodeForDistributedTraining(
        train_x,
        train_y,
        rehash=3000,
        rebuild=10000,
        verbose=False,
        batch_size=10,
    )

    train_x, train_y = gen_training_data()
    test_x, test_y = gen_training_data(n_samples=100)

    untrained_network = build_simple_bolt_network()

    num_of_batches = untrained_network.prepareNodeForDistributedTraining(
        train_x,
        train_y,
        rehash=3000,
        rebuild=10000,
        verbose=False,
        batch_size=10,
    )

    untrained_network.set_weights(
        layer_index=0, new_weights=network.get_weights(layer_index=0)
    )
    untrained_network.set_biases(
        layer_index=0, new_biases=network.get_biases(layer_index=0)
    )
    untrained_network.set_weights(
        layer_index=1, new_weights=network.get_weights(layer_index=1)
    )
    untrained_network.set_biases(
        layer_index=1, new_biases=network.get_biases(layer_index=1)
    )

    train_multiple_networks_same_gradients(
        network, untrained_network, num_of_batches, test_x, test_y
    )
    train_multiple_networks_same_gradients(
        untrained_network, network, num_of_batches, test_x, test_y
    )
