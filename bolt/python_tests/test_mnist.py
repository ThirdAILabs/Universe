import os
from thirdai import bolt, dataset
import sys
import pytest
from collections import namedtuple


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


def train_mnist_sparse_output_layer(args):
    layers = [
        bolt.LayerConfig(dim=256, activation_function=bolt.ActivationFunctions.ReLU),
        bolt.LayerConfig(
            dim=10,
            load_factor=args.sparsity,
            activation_function=bolt.ActivationFunctions.Softmax,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=args.hashes_per_table,
                num_tables=args.num_tables,
                range_pow=args.hashes_per_table * 3,
                reservoir_size=10,
            ),
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=784)

    train_data = dataset.load_bolt_svm_dataset(args.train, 250)
    test_data = dataset.load_bolt_svm_dataset(args.test, 250)
    epoch_times = []
    epoch_accuracies = []
    for _ in range(args.epochs):
        times = network.train(
            train_data,
            bolt.CategoricalCrossEntropyLoss(),
            args.lr,
            1,
            rehash=3000,
            rebuild=10000,
            metrics=[],
            verbose=False,
        )
        epoch_times.append(times["epoch_times"][0])
        acc, _ = network.predict(
            test_data, metrics=["categorical_accuracy"], verbose=False
        )
        epoch_accuracies.append(acc["categorical_accuracy"][0])

    return epoch_accuracies[-1], epoch_accuracies, epoch_times


def train_mnist_sparse_hidden_layer(args):
    layers = [
        bolt.LayerConfig(
            dim=20000,
            load_factor=args.sparsity,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=args.hashes_per_table,
                num_tables=args.num_tables,
                range_pow=args.hashes_per_table * 3,
                reservoir_size=32,
            ),
        ),
        bolt.LayerConfig(dim=10, activation_function=bolt.ActivationFunctions.Softmax),
    ]
    network = bolt.Network(layers=layers, input_dim=784)

    train_data = dataset.load_bolt_svm_dataset(args.train, 250)
    test_data = dataset.load_bolt_svm_dataset(args.test, 250)
    epoch_times = []
    epoch_accuracies = []
    for _ in range(args.epochs):
        times = network.train(
            train_data,
            bolt.CategoricalCrossEntropyLoss(),
            args.lr,
            1,
            rehash=3000,
            rebuild=10000,
            metrics=[],
            verbose=False,
        )
        epoch_times.append(times["epoch_times"][0])
        acc, _ = network.predict(
            test_data, metrics=["categorical_accuracy"], verbose=False
        )
        epoch_accuracies.append(acc["categorical_accuracy"][0])
    return epoch_accuracies[-1], epoch_accuracies, epoch_times


def train(
    args,
    train_fn,
    accuracy_threshold,
    epoch_time_threshold=100,
    total_time_threshold=10000,
):
    final_accuracies = []
    final_epoch_times = []
    total_times = []

    for _ in range(args.runs):

        final_accuracy, accuracies_per_epoch, time_per_epoch = train_fn(args)
        final_accuracies.append(final_accuracy)
        final_epoch_times.append(time_per_epoch[-1])
        total_times.append(sum(time_per_epoch))

        print(
            f"Result of training {args.dataset} for {args.epochs} epochs:\n\tFinal epoch accuracy: {final_accuracy}\n\tFinal epoch time: {time_per_epoch}"
        )

        assert final_accuracies[-1] > accuracy_threshold
        assert final_epoch_times[-1] < epoch_time_threshold
        assert total_times[-1] < total_time_threshold

    return final_accuracies, final_epoch_times


@pytest.mark.integration
def test_mnist_sparse_output():
    args = {
        "dataset": "mnist",
        "train": "mnist",
        "test": "mnist.t",
        "epochs": 10,
        "hashes_per_table": 1,
        "num_tables": 32,
        "sparsity": 0.4,
        "lr": 0.0001,
        "enable_checks": True,
        "runs": 1,
    }
    # Turn the dictionary into the format expected by the train method, fields
    # referencable by e.g. args.train
    args = namedtuple("args", args.keys())(*args.values())
    accs, times = train(
        args, train_fn=train_mnist_sparse_output_layer, accuracy_threshold=0.95
    )


@pytest.mark.integration
def test_mnist_sparse_hidden():
    args = {
        "dataset": "mnist",
        "train": "mnist",
        "test": "mnist.t",
        "epochs": 10,
        "hashes_per_table": 3,
        "num_tables": 64,
        "sparsity": 0.01,
        "lr": 0.0001,
        "enable_checks": True,
        "runs": 1,
    }
    args = namedtuple("args", args.keys())(*args.values())
    accs, times = train(
        args, train_fn=train_mnist_sparse_hidden_layer, accuracy_threshold=0.95
    )


@pytest.mark.integration
def test_load_save_fc_network():
    layers = [
        bolt.LayerConfig(
            dim=1000,
            load_factor=0.2,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=3,
                num_tables=64,
                range_pow=9,
                reservoir_size=32,
            ),
        ),
        bolt.LayerConfig(dim=10, activation_function=bolt.ActivationFunctions.Softmax),
    ]
    network = bolt.Network(layers=layers, input_dim=784)

    train_data = dataset.load_bolt_svm_dataset("mnist", 250)
    test_data = dataset.load_bolt_svm_dataset("mnist.t", 250)

    times = network.train(
        train_data,
        bolt.CategoricalCrossEntropyLoss(),
        0.0001,
        2,
        rehash=3000,
        rebuild=10000,
        metrics=[],
        verbose=False,
    )

    original_acc, _ = network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )

    save_loc = "./bolt_model_save"

    if os.path.exists(save_loc):
        os.remove(save_loc)

    network.save(save_loc)

    new_network = bolt.Network.load(save_loc)

    new_acc, _ = new_network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )

    assert new_acc["categorical_accuracy"][0] == original_acc["categorical_accuracy"][0]

    new_network.train(
        train_data,
        bolt.CategoricalCrossEntropyLoss(),
        0.0001,
        2,
        rehash=3000,
        rebuild=10000,
        metrics=[],
        verbose=False,
    )

    another_acc, _ = new_network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )

    assert another_acc["categorical_accuracy"][0] >= new_acc["categorical_accuracy"][0]

    os.remove(save_loc)
