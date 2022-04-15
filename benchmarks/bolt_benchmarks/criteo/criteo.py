import argparse

from helpers import add_arguments
from mlflow_logger import ExperimentLogger
from thirdai import bolt, dataset


def _define_network(args):
    bottom_mlp = [
        bolt.LayerConfig(
            dim=1000,
            load_factor=0.2,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=3, num_tables=128, range_pow=9, reservoir_size=10
            ),
        ),
        bolt.LayerConfig(dim=100, activation_function=bolt.ActivationFunctions.ReLU),
    ]
    embedding = bolt.EmbeddingLayerConfig(
        num_embedding_lookups=8, lookup_size=16, log_embedding_block_size=10
    )
    top_mlp = [
        bolt.LayerConfig(dim=100, activation_function=bolt.ActivationFunctions.ReLU),
        bolt.LayerConfig(
            dim=1000,
            load_factor=0.2,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=3, num_tables=128, range_pow=9, reservoir_size=10
            ),
        ),
        bolt.LayerConfig(dim=1, activation_function=bolt.ActivationFunctions.Linear),
    ]
    network = bolt.DLRM(embedding, bottom_mlp, top_mlp, 15)
    return network


def train_criteo(args, mlflow_logger):
    network = _define_network(args)
    train_data = dataset.load_click_through_dataset(args.train, 256, 15, 26, True)
    test_data = dataset.load_click_through_dataset(args.test, 256, 15, 26, True)

    mlflow_logger.log_start_training()
    for _ in range(args.epochs):
        network.train(
            train_data,
            loss_fn=bolt.MeanSquaredError(),
            learning_rate=args.lr,
            epochs=1,
            rehash=6400,
            rebuild=128000,
        )
        acc, _ = network.predict(
            test_data, metrics=["categorical_accuracy"], verbose=False
        )
        mlflow_logger.log_epoch(acc["categorical_accuracy"][0])

    acc, _ = network.predict(test_data, metrics=["categorical_accuracy"], verbose=False)
    mlflow_logger.log_epoch(acc["categorical_accuracy"][0])


def main():

    parser = argparse.ArgumentParser(
        description=f"Run BOLT on Criteo with specified params."
    )

    args = add_arguments(
        parser=parser,
        train="/data/criteo/train_shuf.txt",
        test="/data/criteo/test_shuf.txt",
        epochs=5,
        hashes_per_table=5,
        num_tables=128,
        sparsity=0.005,
        lr=0.0001,
    )

    with ExperimentLogger(
        experiment_name="Criteo Click Prediction",
        dataset="criteo",
        algorithm="DLRM",
        framework="bolt",
    ) as mlflow_logger:
        train_criteo(args, mlflow_logger)


if __name__ == "__main__":
    main()
