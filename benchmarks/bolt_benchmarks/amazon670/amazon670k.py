import argparse

from helpers import add_arguments
from mlflow_logger import ExperimentLogger
from thirdai import bolt, dataset


def _define_network(args):
    layers = [
        bolt.FullyConnected(dim=256, activation_function=bolt.ActivationFunctions.ReLU),
        bolt.FullyConnected(
            dim=670091,
            load_factor=args.sparsity,
            activation_function=bolt.ActivationFunctions.Softmax,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=args.hashes_per_table,
                num_tables=args.num_tables,
                range_pow=args.hashes_per_table * 3,
                reservoir_size=128,
            ),
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=135909)
    return network


def train_amzn670(args, mlflow_logger):
    network = _define_network(args)

    train_data = dataset.load_bolt_svm_dataset(args.train, 256)
    test_data = dataset.load_bolt_svm_dataset(args.test, 256)

    mlflow_logger.log_start_training()

    for _ in range(args.epochs):
        network.train(
            train_data,
            bolt.CategoricalCrossEntropyLoss(),
            args.lr,
            epochs=1,
            rehash=6400,
            rebuild=128000,
        )
        acc, __ = network.predict(
            test_data, metrics=["categorical_accuracy"], verbose=False
        )
        mlflow_logger.log_epoch(acc["categorical_accuracy"][0])

    final_accuracy, __ = network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )
    mlflow_logger.log_epoch(final_accuracy["categorical_accuracy"][0])


def main():

    parser = argparse.ArgumentParser(
        description=f"Run BOLT on Amazon 670k with specified params."
    )

    # TODO(vihan) replace these hard-coded paths
    # TODO(vihan) Replace args namespace with a dictionary
    args = add_arguments(
        parser=parser,
        train="/data/amazon-670k/train_shuffled_noHeader.txt",
        test="/data/amazon-670k/test_shuffled_noHeader_sampled.txt",
        epochs=10,
        hashes_per_table=5,
        num_tables=128,
        sparsity=0.005,
        lr=0.0001,
    )

    with ExperimentLogger(
        experiment_name="Product Recommendation",
        dataset="amazon670k",
        algorithm="feedforward",
        framework="bolt",
        experiment_args=args,
    ) as mlflow_logger:
        train_amzn670(args, mlflow_logger)


if __name__ == "__main__":
    main()
