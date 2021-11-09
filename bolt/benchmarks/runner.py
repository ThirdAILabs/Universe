from thirdai import bolt
import sys
import argparse
from helpers import add_arguments, train_and_verify


def train_mnist_sparse_output_layer(args):
    layers = [
        bolt.LayerConfig(dim=256, activation_function="ReLU"),
        bolt.LayerConfig(dim=10, load_factor=args.sparsity,
                         activation_function="Softmax",
                         sampling_config=bolt.SamplingConfig(
                             hashes_per_table=args.K, num_tables=args.L,
                             range_pow=args.K * 3, reservoir_size=10))
    ]
    network = bolt.Network(layers=layers, input_dim=784)
    network.Train(batch_size=250, train_data=args.train, test_data=args.test,
                  learning_rate=args.lr, epochs=args.epochs, rehash=3000, rebuild=10000, max_test_batches=40)
    return network.GetFinalTestAccuracy(), network.GetAccuracyPerEpoch(), network.GetTimePerEpoch()


def train_mnist_sparse_hidden_layer(args):
    layers = [
        bolt.LayerConfig(dim=20000, load_factor=args.sparsity,
                         activation_function="ReLU",
                         sampling_config=bolt.SamplingConfig(
                             hashes_per_table=args.K, num_tables=args.L,
                             range_pow=args.K * 3, reservoir_size=32)),
        bolt.LayerConfig(dim=10, activation_function="Softmax")
    ]
    network = bolt.Network(layers=layers, input_dim=784)
    network.Train(batch_size=250, train_data=args.train, test_data=args.test,
                  learning_rate=args.lr, epochs=args.epochs, rehash=3000, rebuild=10000, max_test_batches=40)
    return network.GetFinalTestAccuracy(), network.GetAccuracyPerEpoch(), network.GetTimePerEpoch()

def train_amzn670(args):
    layers = [
        bolt.LayerConfig(dim=256, activation_function="ReLU"),
        bolt.LayerConfig(dim=670091, load_factor=args.sparsity,
                        activation_function="Softmax",
                        sampling_config=bolt.SamplingConfig(
                            hashes_per_table=args.K, num_tables=args.L,
                            range_pow=args.K * 3, reservoir_size=128)),
    ]
    network = bolt.Network(layers=layers, input_dim=135909)
    network.Train(batch_size=256, train_data=args.train, test_data=args.test, 
                  learning_rate=args.lr, epochs=args.epochs, rehash=6400, rebuild=128000, max_test_batches=20)
    return network.GetFinalTestAccuracy(), network.GetAccuracyPerEpoch(), network.GetTimePerEpoch()


def main():
    assert len(sys.argv) >= 2, \
        "Invalid args, usage: python3 runner.py <dataset> [-h] <flags...>"
    dataset = sys.argv[1]

    # Add default params for training, which can also be specified in command line.
    accuracy_threshold = 0
    max_runs = 1
    train_fn = None
    parser = argparse.ArgumentParser(description=f"Run BOLT on {dataset} with specified params.")
    if dataset == "mnist_so":   
        args = add_arguments(
            parser=parser,
            train="/media/scratch/data/mnist/mnist",
            test="/media/scratch/data/mnist/mnist.t",
            epochs=10,
            K=1,
            L=32,
            sparsity=0.4,
            lr=0.0001,
        )
        train_and_verify(args, train_fn=train_mnist_sparse_output_layer, accuracy_threshold=0.95, max_runs=5)
    elif dataset == "mnist_sh":
        args = add_arguments(
            parser=parser,
            train="/media/scratch/data/mnist/mnist",
            test="/media/scratch/data/mnist/mnist.t",
            epochs=10,
            K=3,
            L=64,
            sparsity=0.01,
            lr=0.0001,
        )
        train_and_verify(args, train_fn=train_mnist_sparse_hidden_layer, accuracy_threshold=0.95, max_runs=5)
    elif dataset == "amzn670":
        args = add_arguments(
            parser=parser,
            train="/media/scratch/data/amazon-670k/train_shuffled_noHeader.txt",
            test="/media/scratch/data/amazon-670k/test_shuffled_noHeader.txt",
            epochs=25,
            K=6,
            L=128,
            sparsity=0.005,
            lr=0.0001,
        )
        train_and_verify(args, train_fn=train_amzn670, accuracy_threshold=0.3, epoch_time_threshold=450, 
                        total_time_threshold=12000, max_runs=1)
    else:
        print("Invalid dataset name. Options: mnist_so, mnist_sh, amzn670, etc.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
