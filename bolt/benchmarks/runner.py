from thirdai import bolt, dataset
import sys
import argparse
from helpers import add_arguments, train


def train_mnist_sparse_output_layer(args):
    layers = [
        bolt.LayerConfig(dim=256, activation_function="ReLU"),
        bolt.LayerConfig(dim=10, load_factor=args.sparsity,
                         activation_function="Softmax",
                         sampling_config=bolt.SamplingConfig(
                             hashes_per_table=args.hashes_per_table, num_tables=args.num_tables,
                             range_pow=args.hashes_per_table * 3, reservoir_size=10))
    ]
    network = bolt.Network(layers=layers, input_dim=784)

    train_data = dataset.loadSVMDataset(args.train, 250)
    test_data = dataset.loadSVMDataset(args.test, 250)
    epoch_times = []
    epoch_accuracies = []
    for _ in range(args.epochs):
        times = network.TrainSparse(train_data, args.lr, 1,
                              rehash=3000, rebuild=10000)
        epoch_times.append(times[0])
        acc = network.TestSparse(test_data)
        epoch_accuracies.append(acc)
    return epoch_accuracies[-1], epoch_accuracies, epoch_times


def train_mnist_sparse_hidden_layer(args):
    layers = [
        bolt.LayerConfig(dim=20000, load_factor=args.sparsity,
                         activation_function="ReLU",
                         sampling_config=bolt.SamplingConfig(
                             hashes_per_table=args.hashes_per_table, num_tables=args.num_tables,
                             range_pow=args.hashes_per_table * 3, reservoir_size=32)),
        bolt.LayerConfig(dim=10, activation_function="Softmax")
    ]
    network = bolt.Network(layers=layers, input_dim=784)

    train_data = dataset.loadSVMDataset(args.train, 250)
    test_data = dataset.loadSVMDataset(args.test, 250)
    epoch_times = []
    epoch_accuracies = []
    for _ in range(args.epochs):
        times = network.TrainSparse(train_data, args.lr, 1,
                              rehash=3000, rebuild=10000)
        epoch_times.append(times[0])
        acc = network.TestSparse(test_data)
        epoch_accuracies.append(acc)
    return epoch_accuracies[-1], epoch_accuracies, epoch_times


def train_amzn670(args):
    layers = [
        bolt.LayerConfig(dim=256, activation_function="ReLU"),
        bolt.LayerConfig(dim=670091, load_factor=args.sparsity,
                         activation_function="Softmax",
                         sampling_config=bolt.SamplingConfig(
                             hashes_per_table=args.hashes_per_table, num_tables=args.num_tables,
                             range_pow=args.hashes_per_table * 3, reservoir_size=128)),
    ]
    network = bolt.Network(layers=layers, input_dim=135909)

    train_data = dataset.loadSVMDataset(args.train, 256)
    test_data = dataset.loadSVMDataset(args.test, 256)
    epoch_times = []
    epoch_accuracies = []
    for _ in range(args.epochs):
        times = network.TrainSparse(train_data, args.lr, 1,
                              rehash=6400, rebuild=128000)
        epoch_times.append(times[0])
        acc = network.TestSparse(test_data, batch_limit=20)
        epoch_accuracies.append(acc)
    final_accuracy = network.TestSparse(test_data)
    return final_accuracy, epoch_accuracies, epoch_times


def main():
    assert len(sys.argv) >= 2, \
        "Invalid args, usage: python3 runner.py <dataset> [-h] <flags...>"
    dataset = sys.argv[1]

    # Add default params for training, which can also be specified in command line.
    accs, times = [], []
    parser = argparse.ArgumentParser(
        description=f"Run BOLT on {dataset} with specified params.")
    if dataset == "mnist_sparse_output":
        args = add_arguments(
            parser=parser,
            train="/media/scratch/data/mnist/mnist",
            test="/media/scratch/data/mnist/mnist.t",
            epochs=10,
            hashes_per_table=1,
            num_tables=32,
            sparsity=0.4,
            lr=0.0001,
        )
        accs, times = train(
            args, train_fn=train_mnist_sparse_output_layer, accuracy_threshold=0.95)
    elif dataset == "mnist_sparse_hidden":
        args = add_arguments(
            parser=parser,
            train="/media/scratch/data/mnist/mnist",
            test="/media/scratch/data/mnist/mnist.t",
            epochs=10,
            hashes_per_table=3,
            num_tables=64,
            sparsity=0.01,
            lr=0.0001,
        )
        accs, times = train(
            args, train_fn=train_mnist_sparse_hidden_layer, accuracy_threshold=0.95)
    elif dataset == "amzn670":
        args = add_arguments(
            parser=parser,
            train="/media/scratch/data/amazon-670k/train_shuffled_noHeader.txt",
            test="/media/scratch/data/amazon-670k/test_shuffled_noHeader.txt",
            epochs=25,
            hashes_per_table=6,
            num_tables=128,
            sparsity=0.005,
            lr=0.0001,
        )
        accs, times = train(args, train_fn=train_amzn670, accuracy_threshold=0.3, epoch_time_threshold=450,
                            total_time_threshold=12000)
    else:
        print("Invalid dataset name. Options: mnist_sparse_output, mnist_sparse_hidden, amzn670, etc.", file=sys.stderr)
        sys.exit(1)
    print("Avg final accuracies: ", sum(accs) / len(accs))
    print("Avg final epoch time (s): ", sum(times) / len(times))


if __name__ == "__main__":
    main()
