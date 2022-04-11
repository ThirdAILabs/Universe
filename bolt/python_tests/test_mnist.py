import os
from thirdai import bolt, dataset
import pytest

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


# Constructs a bolt network for mnist with a sparse output layer.
def build_sparse_output_layer_network():
    layers = [
        bolt.LayerConfig(
            dim=256, activation_function=bolt.ActivationFunctions.ReLU),
        bolt.LayerConfig(
            dim=10,
            load_factor=0.4,
            activation_function=bolt.ActivationFunctions.Softmax,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=1,
                num_tables=32,
                range_pow=3,
                reservoir_size=10,
            ),
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=784)
    return network


# Constructs a bolt network for mnist with a sparse hidden layer. The parameters dim and sparsity are for this sparse hidden layer.
def build_sparse_hidden_layer_network(dim, sparsity):
    layers = [
        bolt.LayerConfig(
            dim=dim,
            load_factor=sparsity,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=3,
                num_tables=64,
                range_pow=9,
                reservoir_size=32,
            ),
        ),
        bolt.LayerConfig(
            dim=10, activation_function=bolt.ActivationFunctions.Softmax),
    ]
    network = bolt.Network(layers=layers, input_dim=784)
    return network


def train_network(network, train_data, epochs, learning_rate=LEARNING_RATE):
    times = network.train(
        train_data,
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
    train_data = dataset.load_bolt_svm_dataset("mnist", 250)
    test_data = dataset.load_bolt_svm_dataset("mnist.t", 250)
    return train_data, test_data


ACCURACY_THRESHOLD = 0.94
SPARSE_INFERENCE_ACCURACY_THRESHOLD = 0.9


@pytest.mark.integration
def test_mnist_sparse_output_layer():
    network = build_sparse_output_layer_network()

    train, test = load_mnist()

    train_network(network, train_data=train, epochs=10)

    acc, _ = network.predict(
        test, metrics=["categorical_accuracy"], verbose=False)

    assert acc["categorical_accuracy"][0] >= ACCURACY_THRESHOLD


@pytest.mark.integration
def test_mnist_sparse_hidden_layer():
    network = build_sparse_hidden_layer_network(20000, 0.01)

    train, test = load_mnist()

    train_network(network, train_data=train, epochs=10)

    acc, _ = network.predict(
        test, metrics=["categorical_accuracy"], verbose=False)

    assert acc["categorical_accuracy"][0] >= ACCURACY_THRESHOLD


@pytest.mark.integration
def test_mnist_sparse_inference():
    network = build_sparse_hidden_layer_network(20000, 0.01)

    train, test = load_mnist()

    train_network(network, train_data=train, epochs=9)

    dense_predict, _ = network.predict(
        test, metrics=["categorical_accuracy"], verbose=False
    )

    assert dense_predict["categorical_accuracy"][0] >= ACCURACY_THRESHOLD

    network.enable_sparse_inference()

    train_network(network, train_data=train, epochs=1)

    sparse_predict, _ = network.predict(
        test, metrics=["categorical_accuracy"], verbose=False
    )

    assert (
        sparse_predict["categorical_accuracy"][0] >= SPARSE_INFERENCE_ACCURACY_THRESHOLD
    )

    dense_time = dense_predict["test_time"][0]
    sparse_time = sparse_predict["test_time"][0]

    SPARSE_INFERENCE_SPEED_MULTIPLIER = 5

    assert (sparse_time * SPARSE_INFERENCE_SPEED_MULTIPLIER) < dense_time


@pytest.mark.integration
def test_load_save_fc_network():
    network = build_sparse_hidden_layer_network(1000, 0.2)

    train_data, test_data = load_mnist()

    train_network(network, train_data=train_data, epochs=2)

    original_acc, _ = network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )

    save_loc = "./bolt_model_save"

    if os.path.exists(save_loc):
        os.remove(save_loc)

    # Save network and load as a new network
    network.save(save_loc)

    new_network = bolt.Network.load(save_loc)

    new_acc, _ = new_network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )

    assert new_acc["categorical_accuracy"][0] == original_acc["categorical_accuracy"][0]

    # Continue to train loaded network
    train_network(new_network, train_data=train_data, epochs=2)

    another_acc, _ = new_network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )

    assert another_acc["categorical_accuracy"][0] >= new_acc["categorical_accuracy"][0]

    os.remove(save_loc)


@pytest.mark.integration
def test_get_set_weights():
    network = build_sparse_output_layer_network()

    train_data, test_data = load_mnist()

    train_network(network, train_data=train_data, epochs=10)

    original_acc, _ = network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )

    assert original_acc["categorical_accuracy"][0] >= ACCURACY_THRESHOLD

    untrained_network = build_sparse_output_layer_network()

    untrained_network.set_weights(0, network.get_weights(0))
    untrained_network.set_weights(1, network.get_weights(1))

    new_acc, _ = untrained_network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )

    assert new_acc["categorical_accuracy"][0] >= ACCURACY_THRESHOLD
