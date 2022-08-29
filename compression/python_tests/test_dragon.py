import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]

import numpy as np
from thirdai import bolt

from utils import (
    gen_numpy_training_data,
    train_single_node_distributed_network,
    build_simple_distributed_bolt_network,
)

INPUT_DIM = 10
HIDDEN_DIM = 10
OUTPUT_DIM = 10
LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8

# A compressed dragon vector is a dictionary at the moment.
# It has the following keys: "compression_scheme", "original_size", "sketch_size"
# "seed_for_hashing", "compression_density", "indices", "values"


def get_compressed_dragon_gradients(
    network, num_layers, compression_density, seed_for_hashing
):
    compressed_weight_grads = []

    for layer_index in range(num_layers):
        compressed_weight_grads.append(
            network.get_compressed_weight_gradients(
                compression_scheme="dragon",
                layer_index=layer_index,
                compression_density=compression_density,
                seed_for_hashing=seed_for_hashing,
            )
        )
    return compressed_weight_grads


def set_compressed_dragon_gradients(network, num_layers, compressed_weight_grads):
    for layer_index in range(num_layers):
        network.set_compressed_weight_gradients(
            layer_index=layer_index,
            compressed_vector=compressed_weight_grads[layer_index],
        )
    return network


# We will get a compressed vector of gradients and then check whether the values are right
def test_get_gradients():
    network = build_simple_distributed_bolt_network(
        input_dim=INPUT_DIM, sparse_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, sparsity=1.0
    )

    train_data, train_labels = gen_numpy_training_data(n_classes=10, n_samples=1000)

    train_single_node_distributed_network(
        network, train_data, train_labels, epochs=1, update_parameters=False
    )

    first_layer_weight_gradients = np.ravel(
        network.get_weights_gradients(layer_index=0)
    )

    first_layer_biases_gradients = np.ravel(network.get_biases_gradients(layer_index=0))

    # getting the compressed gradients
    compressed_weight_gradients = network.get_compressed_weight_gradients(
        compression_scheme="dragon",
        layer_index=0,
        compression_density=0.2,
        seed_for_hashing=1,
    )

    compressed_biases_gradients = network.get_compressed_biases_gradients(
        compression_scheme="dragon",
        layer_index=0,
        compression_density=0.2,
        seed_for_hashing=1,
    )

    # checking whether the gradients are correct
    for i, indices in enumerate(compressed_weight_gradients["indices"]):
        if indices != 0:
            assert (
                first_layer_weight_gradients[indices]
                == compressed_weight_gradients["values"][i]
            )

    for i, indices in enumerate(compressed_biases_gradients["indices"]):
        if indices != 0:
            assert (
                first_layer_biases_gradients[indices]
                == compressed_biases_gradients["values"][i]
            )

    assert (
        compressed_weight_gradients["original_size"]
        == first_layer_weight_gradients.shape[0]
    )
    assert (
        compressed_biases_gradients["original_size"]
        == first_layer_biases_gradients.shape[0]
    )

    # setting the compressed gradients
    network.set_compressed_weight_gradients(
        layer_index=0, compressed_vector=compressed_weight_gradients
    )

    network.set_compressed_biases_gradients(
        layer_index=0, compressed_vector=compressed_biases_gradients
    )


def test_set_gradients():
    network = build_simple_distributed_bolt_network(
        input_dim=INPUT_DIM, sparse_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, sparsity=1.0
    )

    train_data, train_labels = gen_numpy_training_data(n_classes=10, n_samples=1000)

    train_single_node_distributed_network(
        network, train_data, train_labels, epochs=1, update_parameters=False
    )

    # getting the compressed gradients
    compressed_weight_gradients = network.get_compressed_weight_gradients(
        compression_scheme="dragon",
        layer_index=0,
        compression_density=0.2,
        seed_for_hashing=1,
    )

    compressed_biases_gradients = network.get_compressed_biases_gradients(
        compression_scheme="dragon",
        layer_index=0,
        compression_density=0.2,
        seed_for_hashing=1,
    )

    reconstruced_weight_gradients = np.ravel(
        network.get_weights_gradients(layer_index=0)
    )

    reconstruced_biases_gradients = np.ravel(
        network.get_biases_gradients(layer_index=0)
    )

    # checking whether the gradients are correct
    for i, indices in enumerate(compressed_weight_gradients["indices"]):
        if indices != 0:
            assert (
                reconstruced_weight_gradients[indices]
                == compressed_weight_gradients["values"][i]
            )

    for i, indices in enumerate(compressed_biases_gradients["indices"]):
        if indices != 0:
            assert (
                reconstruced_biases_gradients[indices]
                == compressed_biases_gradients["values"][i]
            )


def test_compressed_training():
    network = build_simple_distributed_bolt_network(
        input_dim=INPUT_DIM, sparse_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, sparsity=1.0
    )

    train_data, train_labels = gen_numpy_training_data(n_classes=10, n_samples=1000)
    test_data, test_labels = gen_numpy_training_data(n_classes=10, n_samples=100)

    batch_size = network.prepareNodeForDistributedTraining(
        train_data,
        train_labels,
        rehash=3000,
        rebuild=10000,
        verbose=True,
    )

    for epochs in range(20):
        for batch_num in range(batch_size):
            network.calculateGradientSingleNode(
                batch_num, bolt.CategoricalCrossEntropyLoss()
            )
            compressed_weight_grads = get_compressed_dragon_gradients(
                network,
                num_layers=2,
                compression_density=0.5,
                seed_for_hashing=np.random.randint(100),
            )
            network = set_compressed_dragon_gradients(
                network, num_layers=2, compressed_weight_grads=compressed_weight_grads
            )
            network.updateParametersSingleNode(LEARNING_RATE)

    acc, _ = network.predictSingleNode(
        test_data, test_labels, metrics=["categorical_accuracy"], verbose=False
    )
    assert acc["categorical_accuracy"] >= ACCURACY_THRESHOLD


# test_get_gradients()
# test_set_gradients()
# test_compressed_training()
