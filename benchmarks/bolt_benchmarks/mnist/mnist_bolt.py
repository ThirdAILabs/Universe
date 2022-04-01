from thirdai import bolt, dataset
import sys
import argparse
from helpers import add_arguments
import requests
import time
from datetime import date
import numpy as np

# Add the logging folder to the system path
import sys

# sys.path.insert(1, sys.path[0] + "/../../logging/")
from mlflow_logger import ExperimentLogger


def _define_network(args):
    layers = [
        bolt.LayerConfig(
            dim=256,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.LayerConfig(dim=10, activation_function=bolt.ActivationFunctions.Softmax),
    ]

    network = bolt.Network(layers=layers, input_dim=784)
    return network


def train_birds(args, network, mlflow_logger):
    train_data = dataset.load_bolt_svm_dataset(args.train, 256)
    test_data = dataset.load_bolt_svm_dataset(args.test, 256)

    mlflow_logger.log_start_training()
    for _ in range(args.epochs):
        # TODO(vihan) Add a default batch size to the train() function
        # signature to avoid specifying it here.
        network.train(
            train_data,
            loss_fn=bolt.CategoricalCrossEntropyLoss(),
            learning_rate=args.lr,
            epochs=1,
            rehash=3000,
            rebuild=10000,
        )
        acc, __ = network.predict(
            test_data,
            metrics=["categorical_accuracy"],
            verbose=False,
        )
        mlflow_logger.log_epoch(acc["categorical_accuracy"][0])

    final_accuracy, _ = network.predict(
        test_data,
        metrics=["categorical_accuracy"],
        verbose=False,
    )

    print(final_accuracy)
    mlflow_logger.log_final_accuracy(final_accuracy["categorical_accuracy"][0])


def main():

    parser = argparse.ArgumentParser(
        description=f"Run BOLT on Birds-400 with specified params."
    )

    # TODO(vihan): Fix the train/test paths for numpy inputs
    args = add_arguments(
        parser=parser,
        train="/share/data/mnist/mnist",
        test="/share/data/mnist/mnist.t",
        epochs=100,
        hashes_per_table=4,
        num_tables=64,
        sparsity=0.05,
        lr=0.0001,
    )

    layers = [
        bolt.LayerConfig(
            dim=3000,
            activation_function=bolt.ActivationFunctions.ReLU,
            load_factor=0.05,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=4, num_tables=64, range_pow=14, reservoir_size=32
            ),
        ),
        bolt.LayerConfig(dim=100, activation_function=bolt.ActivationFunctions.Softmax),
    ]

    network = bolt.Network(layers=layers, input_dim=1536)

    network = _define_network(args)

    with ExperimentLogger(
        experiment_name="MNIST Benchmark",
        dataset="mnist",
        algorithm="Bolt",
    ) as mlflow_logger:
        train_birds(args, network, mlflow_logger)


if __name__ == "__main__":
    main()
