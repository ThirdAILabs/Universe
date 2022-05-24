# Add an integration test marker for all tests in this file
import pytest

pytestmark = [pytest.mark.integration]

import os
from thirdai import bolt, dataset
import numpy as np

LEARNING_RATE = 0.0001


def setup_module():
    if not os.path.exists("mnist"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")

    if not os.path.exists("mnist.t"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")


def load_mnist_labels():
    labels = []
    with open("mnist.t") as file:
        for line in file.readlines():
            label = int(line.split(" ")[0])
            labels.append(label)
    return np.array(labels)


# Constructs a bolt network for mnist with a sparse output layer.
def build_sparse_output_layer_network():
    layers = [
        bolt.FullyConnected(dim=256, activation_function=bolt.ActivationFunctions.ReLU),
        bolt.FullyConnected(
            dim=10,
            load_factor=0.4,
            activation_function=bolt.ActivationFunctions.Softmax,
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=784)
    return network


# Constructs a bolt network for mnist with a sparse hidden layer. The parameters dim and sparsity are for this sparse hidden layer.
def build_sparse_hidden_layer_network(dim, sparsity):
    layers = [
        bolt.FullyConnected(
            dim=dim,
            load_factor=sparsity,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(
            dim=10, activation_function=bolt.ActivationFunctions.Softmax
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=784)
    return network


def train_network(
    network, train_data, train_labels, epochs, learning_rate=LEARNING_RATE
):
    times = network.train(
        train_data,
        train_labels,
        bolt.CategoricalCrossEntropyLoss(),
        learning_rate,
        epochs,
        rehash=3000,
        rebuild=10000,
        metrics=[],
        verbose=False,
    )
    return times


def load_mnist():
    train_x, train_y = dataset.load_bolt_svm_dataset("mnist", 250)
    test_x, test_y = dataset.load_bolt_svm_dataset("mnist.t", 250)
    return train_x, train_y, test_x, test_y


ACCURACY_THRESHOLD = 0.94
SPARSE_INFERENCE_ACCURACY_THRESHOLD = 0.9
SPARSE_INFERENCE_SPARSE_OUTPUT_ACCURACY_THRESHOLD = 0.35


def test_mnist_sparse_output_layer():
    network = build_sparse_output_layer_network()

    train_x, train_y, test_x, test_y = load_mnist()

    train_network(network, train_x, train_y, epochs=10)

    acc, activations = network.predict(
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


def test_mnist_sparse_hidden_layer():
    network = build_sparse_hidden_layer_network(20000, 0.01)

    train_x, train_y, test_x, test_y = load_mnist()

    train_network(network, train_x, train_y, epochs=10)

    acc, activations = network.predict(
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


def test_mnist_sparse_inference():
    network = build_sparse_hidden_layer_network(20000, 0.01)

    train_x, train_y, test_x, test_y = load_mnist()

    train_network(network, train_x, train_y, epochs=9)

    dense_predict, _ = network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert dense_predict["categorical_accuracy"] >= ACCURACY_THRESHOLD

    network.enable_sparse_inference()

    train_network(network, train_x, train_y, epochs=1)

    sparse_predict, _ = network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert sparse_predict["categorical_accuracy"] >= SPARSE_INFERENCE_ACCURACY_THRESHOLD

    dense_time = dense_predict["test_time"]
    sparse_time = sparse_predict["test_time"]

    SPARSE_INFERENCE_SPEED_MULTIPLIER = 5

    assert (sparse_time * SPARSE_INFERENCE_SPEED_MULTIPLIER) < dense_time


# This test will not get great accuracy because the output layer (10 neurons)
# is too small for good sampling.
# However this test makes sure we have a non random level of accuarcy, and also
# tests that the sparse activations returned are corretct.
def test_sparse_inference_with_sparse_output():
    network = build_sparse_output_layer_network()

    train_x, train_y, test_x, test_y = load_mnist()

    train_network(network, train_x, train_y, epochs=10)

    dense_predict, _ = network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert dense_predict["categorical_accuracy"] >= ACCURACY_THRESHOLD

    network.enable_sparse_inference()

    train_network(network, train_x, train_y, epochs=1)

    sparse_predict, active_neurons, activations = network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert (
        sparse_predict["categorical_accuracy"]
        >= SPARSE_INFERENCE_SPARSE_OUTPUT_ACCURACY_THRESHOLD
    )

    # This last check is just to make sure that the accuracy computed in c++ matches
    # what we can compute here using the returned activations. This verifies that the
    # returned activations match and that the metrics are computed correctly.
    argmax_indices = np.argmax(activations, axis=1)
    predictions = active_neurons[np.arange(len(active_neurons)), argmax_indices]

    labels = load_mnist_labels()
    acc_computed = np.mean(predictions == labels)

    assert sparse_predict["categorical_accuracy"] == acc_computed


def test_load_save_fc_network():
    network = build_sparse_hidden_layer_network(1000, 0.2)

    train_x, train_y, test_x, test_y = load_mnist()

    train_network(network, train_x, train_y, epochs=2)

    original_acc, _ = network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    save_loc = "./bolt_model_save"

    if os.path.exists(save_loc):
        os.remove(save_loc)

    # Save network and load as a new network
    network.save(save_loc)

    new_network = bolt.Network.load(save_loc)

    new_acc, _ = new_network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert new_acc["categorical_accuracy"] == original_acc["categorical_accuracy"]

    # Continue to train loaded network
    train_network(new_network, train_x, train_y, epochs=2)

    another_acc, _ = new_network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert another_acc["categorical_accuracy"] >= new_acc["categorical_accuracy"]

    os.remove(save_loc)


def test_get_set_weights():
    network = build_sparse_output_layer_network()

    train_x, train_y, test_x, test_y = load_mnist()

    train_network(network, train_x, train_y, epochs=10)

    original_acc, _ = network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert original_acc["categorical_accuracy"] >= ACCURACY_THRESHOLD

    untrained_network = build_sparse_output_layer_network()

    untrained_network.set_weights(0, network.get_weights(0))
    untrained_network.set_weights(1, network.get_weights(1))

    untrained_network.set_biases(0, network.get_biases(0))
    untrained_network.set_biases(1, network.get_biases(1))

    new_acc, _ = untrained_network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert new_acc["categorical_accuracy"] == original_acc["categorical_accuracy"]
