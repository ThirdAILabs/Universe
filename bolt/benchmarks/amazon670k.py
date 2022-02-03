from thirdai import bolt, dataset
import sys
import argparse
from helpers import add_arguments
import matplotlib.pyplot as plt
import requests
import time
import mlflow
from datetime import date


def train_amzn670(args):
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

    for i in range(args.epochs):
        network.train(train_data, args.lr, 1, rehash=6400, rebuild=128000)
        acc = network.predict(test_data, batch_limit=20)
        mlflow.log_metric("accuracy", acc)

    final_accuracy = network.predict(test_data)
    mlflow.log_metric("final_accuracy", final_accuracy)


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

    mlflow.set_tracking_uri(
        "http://deplo-mlflo-15qe25sw8psjr-1d20dd0c302edb1f.elb.us-east-1.amazonaws.com"
    )
    mlflow.set_experiment("Bolt")
    with mlflow.start_run(
        run_name=f"Amazon670k Benchmarks {date.today()}",
        tags={"dataset": "amazon670k", "algorithm": "bolt"},
    ):
        mlflow.log_param("hashes_per_table", args.hashes_per_table)
        mlflow.log_param("num_tables", args.num_tables)
        mlflow.log_param("sparsity", args.sparsity)
        mlflow.log_param("learning_rate", args.lr)

        train_amzn670(args)


if __name__ == "__main__":
    main()
