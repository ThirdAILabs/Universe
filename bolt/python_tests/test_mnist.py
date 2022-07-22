# Add an integration test marker for all tests in this file
import pytest

pytestmark = [pytest.mark.integration]

from thirdai import bolt, dataset
import numpy as np
import os

from .utils import (
    train_network,
    build_sparse_hidden_layer_classifier,
    copy_two_layer_network_parameters,
)

LEARNING_RATE = 0.0001
ACCURACY_THRESHOLD = 0.94
SPARSE_INFERENCE_ACCURACY_THRESHOLD = 0.9
SPARSE_INFERENCE_SPARSE_OUTPUT_ACCURACY_THRESHOLD = 0.35

# Constructs a bolt network for mnist with a sparse output layer.
def build_sparse_output_layer_network():
    layers = [
        bolt.FullyConnected(dim=256, activation_function="ReLU"),
        bolt.FullyConnected(
            dim=10,
            sparsity=0.4,
            activation_function="Softmax",
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=784)
    return network


def check_categorical_accuracies(acc, activations, accuracy_threshold):

    assert acc["categorical_accuracy"] >= accuracy_threshold  # ACCURACY_THRESHOLD

    # This last check is just to make sure that the accuracy computed in c++ matches
    # what we can compute here using the returned activations. This verifies that the
    # returned activations match and that the metrics are computed correctly.
    predictions = np.argmax(activations, axis=1)

    labels = load_mnist_labels()
    acc_computed = np.mean(predictions == labels)

    assert acc_computed == acc["categorical_accuracy"]

def load_mnist():
    train_x, train_y = dataset.load_bolt_svm_dataset("mnist", 250)
    test_x, test_y = dataset.load_bolt_svm_dataset("mnist.t", 250)
    return train_x, train_y, test_x, test_y


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



def test_mnist_sparse_output_layer():
    network = build_sparse_output_layer_network()

    train_x, train_y, test_x, test_y = load_mnist()

    train_network(network, train_x, train_y, epochs=10)

    acc, activations = network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    check_categorical_accuracies(acc, activations, ACCURACY_THRESHOLD)


def test_mnist_sparse_hidden_layer():
    network = build_sparse_hidden_layer_classifier(
        input_dim=784, sparse_dim=20000, output_dim=10, sparsity=0.01
    )

    train_x, train_y, test_x, test_y = load_mnist()

    train_network(network, train_x, train_y, epochs=12)

    acc, activations = network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    check_categorical_accuracies(acc, activations, ACCURACY_THRESHOLD)


def test_mnist_sparse_inference():
    network = build_sparse_hidden_layer_classifier(
        input_dim=784, sparse_dim=20000, output_dim=10, sparsity=0.01
    )

    train_x, train_y, test_x, test_y = load_mnist()

    train_network(network, train_x, train_y, epochs=9)

    dense_predict, _ = network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert dense_predict["categorical_accuracy"] >= ACCURACY_THRESHOLD

    network.freeze_hash_tables()

    train_network(network, train_x, train_y, epochs=1)

    sparse_predict, _ = network.predict(
        test_x,
        test_y,
        sparse_inference=True,
        metrics=["categorical_accuracy"],
        verbose=False,
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

    network.freeze_hash_tables()

    train_network(network, train_x, train_y, epochs=1)

    sparse_predict, activations, active_neurons = network.predict(
        test_x,
        test_y,
        sparse_inference=True,
        metrics=["categorical_accuracy"],
        verbose=False,
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


def test_get_set_weights():
    network = build_sparse_output_layer_network()

    train_x, train_y, test_x, test_y = load_mnist()

    train_network(network, train_x, train_y, epochs=10)

    original_acc, _ = network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert original_acc["categorical_accuracy"] >= ACCURACY_THRESHOLD

    untrained_network = build_sparse_output_layer_network()

    copy_two_layer_network_parameters(network, untrained_network)

    new_acc, _ = untrained_network.predict(
        test_x, test_y, metrics=["categorical_accuracy"], verbose=False
    )

    assert new_acc["categorical_accuracy"] == original_acc["categorical_accuracy"]
