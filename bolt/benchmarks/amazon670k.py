from thirdai import bolt, dataset
import sys
import argparse
from helpers import add_arguments
import requests
import time
from datetime import date

# Add the logging folder to the system path
import sys

sys.path.insert(1, sys.path[0] + "/../../logging/")
from mlflow_logger import ModelLogger


def train_amzn670(args, mlflow_logger):
    layers = [
        bolt.LayerConfig(dim=256, activation_function="ReLU"),
        bolt.LayerConfig(
            dim=670091,
            load_factor=args.sparsity,
            activation_function="Softmax",
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=args.hashes_per_table,
                num_tables=args.num_tables,
                range_pow=args.hashes_per_table * 3,
                reservoir_size=128,
            ),
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=135909)

    train_data = dataset.load_svm_dataset(args.train, 256)
    test_data = dataset.load_svm_dataset(args.test, 256)

    mlflow_logger.log_start_training()
    for i in range(args.epochs):
        network.train(train_data, args.lr, 1, rehash=6400, rebuild=128000)
        acc = network.predict(test_data, batch_limit=20)
        mlflow.log_epoch(acc)

    final_accuracy = network.predict(test_data)
    mlflow.log_param("final_accuracy", final_accuracy)


def main():

    parser = argparse.ArgumentParser(
        description=f"Run BOLT on Amazon 670k with specified params."
    )

    args = add_arguments(
        parser=parser,
        train="/media/scratch/data/amazon-670k/train_shuffled_noHeader.txt",
        test="/media/scratch/data/amazon-670k/test_shuffled_noHeader.txt",
        epochs=25,
        hashes_per_table=5,
        num_tables=128,
        sparsity=0.005,
        lr=0.0001,
    )

    with ModelLogger(
        dataset="amazon670k",
        learning_rate=0.01,
        num_hash_tables=10,
        hashes_per_table=5,
        sparsity=0.01,
        algorithm="Bolt",
    ) as mlflow_logger:

        train_amzn670(args, mlflow_logger)


if __name__ == "__main__":
    main()
