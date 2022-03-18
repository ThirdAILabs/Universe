import os
import time
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
        bolt.LayerConfig(dim=256, activation_function="ReLU"),
        bolt.LayerConfig(
            dim=10,
            load_factor=args.sparsity,
            activation_function="Softmax",
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=args.hashes_per_table,
                num_tables=args.num_tables,
                range_pow=args.hashes_per_table * 3,
                reservoir_size=10,
            ),
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=784)

    train_data = dataset.load_svm_dataset(args.train, 250)
    test_data = dataset.load_svm_dataset(args.test, 250)
    epoch_times = []
    epoch_accuracies = []
    for _ in range(args.epochs):
        times = network.train(train_data, args.lr, 1, rehash=3000, rebuild=10000)
        epoch_times.append(times[0])
        acc = network.predict(test_data)
        epoch_accuracies.append(acc)

    return epoch_accuracies[-1], epoch_accuracies, epoch_times


def train_mnist_sparse_hidden_layer(args):
    layers = [
        bolt.LayerConfig(
            dim=20000,
            load_factor=args.sparsity,
            activation_function="ReLU",
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=args.hashes_per_table,
                num_tables=args.num_tables,
                range_pow=args.hashes_per_table * 3,
                reservoir_size=32,
            ),
        ),
        bolt.LayerConfig(dim=10, activation_function="Softmax"),
    ]
    network = bolt.Network(layers=layers, input_dim=784)

    train_data = dataset.load_svm_dataset(args.train, 250)
    test_data = dataset.load_svm_dataset(args.test, 250)
    epoch_times = []
    epoch_accuracies = []
    for _ in range(args.epochs):
        times = network.train(train_data, args.lr, 1, rehash=3000, rebuild=10000)
        epoch_times.append(times[0])
        acc = network.predict(test_data)
        epoch_accuracies.append(acc)
    return epoch_accuracies[-1], epoch_accuracies, epoch_times

def train_mnist_sparse_hidden_inference(args):
    layers = [
        bolt.LayerConfig(
            dim=20000,
            load_factor=args.sparsity,
            activation_function="ReLU",
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=args.hashes_per_table,
                num_tables=args.num_tables,
                range_pow=args.hashes_per_table * 3,
                reservoir_size=32,
            ),
        ),
        bolt.LayerConfig(dim=10, activation_function="Softmax"),
    ]
    network = bolt.Network(layers=layers, input_dim=784)

    train_data = dataset.load_svm_dataset(args.train, 250)
    test_data = dataset.load_svm_dataset(args.test, 250)
    epoch_times = []
    inf_times = []
    epoch_accuracies = []
    for i in range(args.epochs):
        if(i == (args.epochs - 1)):
           network.use_sparse_inference()
        times = network.train(train_data, args.lr, 1, rehash=3000, rebuild=10000)
        epoch_times.append(times[0])
        t0 = time.time()
        acc = network.predict(test_data)
        t1 = time.time()
        inf_times.append(t1-t0)
        epoch_accuracies.append(acc)
    return epoch_accuracies[-1], epoch_accuracies, epoch_times, inf_times


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

def train_with_inference(
    args,
    train_fn,
    accuracy_threshold,
    speed_multiplier_threshold=5,
    epoch_time_threshold=100,
    total_time_threshold=10000,
):
    final_accuracies = []
    final_epoch_times = []
    total_times = []

    for _ in range(args.runs):

        final_accuracy, accuracies_per_epoch, time_per_epoch, inf_times = train_fn(args)
        final_accuracies.append(final_accuracy)
        final_epoch_times.append(time_per_epoch[-1])
        total_times.append(sum(time_per_epoch))

        print(
            f"Result of training {args.dataset} for {args.epochs} epochs:\n\tFinal epoch accuracy: {final_accuracy}\n\tFinal epoch time: {time_per_epoch}"
        )

        assert final_accuracies[-1] > accuracy_threshold
        assert final_epoch_times[-1] < epoch_time_threshold
        assert total_times[-1] < total_time_threshold
        avg_non_inf = sum(inf_times[:-1])/(args.epochs - 1)
        # print(avg_non_inf, inf_times[-1])
        assert avg_non_inf >= inf_times[-1] * speed_multiplier_threshold

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
def test_mnist_sparse_inference():
    args = {
    "dataset": "mnist",
    "train": "mnist",
    "test": "mnist.t",
    "epochs": 4,
    "hashes_per_table": 3,
    "num_tables": 64,
    "sparsity": 0.01,
    "lr": 0.0001,
    "enable_checks": True,
    "runs": 1,
    }   
    args = namedtuple("args", args.keys())(*args.values())

    accs, times = train_with_inference(
        args, train_fn=train_mnist_sparse_hidden_inference, accuracy_threshold=0.90, speed_multiplier_threshold=5
    )
