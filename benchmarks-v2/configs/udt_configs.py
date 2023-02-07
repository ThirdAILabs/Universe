import json

import numpy as np
from configs import UDTBenchmarkConfig
from thirdai import bolt, deployment


class YelpPolarityUDTConfig(UDTBenchmarkConfig):
    train_file = "/share/data/udt_datasets/yelp_polarity/train.csv"
    test_file = "/share/data/udt_datasets/yelp_polarity/test.csv"

    data_types = {
        "text": bolt.types.text(),
        "label": bolt.types.categorical(),
    }

    target = "label"
    delimiter = "\t"
    experiment_name = "YelpPolarityUDT"
    dataset_name = "yelp_polarity"


class AmazonPolarityUDTConfig(UDTBenchmarkConfig):
    experiment_name = "AmazonPolarityUDT"
    dataset_name = "amazon_polarity"
    train_file = (
        "/share/data/udt_datasets/amazon_polarity/amazon_polarity_content_train.csv"
    )
    test_file = (
        "/share/data/udt_datasets/amazon_polarity/amazon_polarity_content_test.csv"
    )

    data_types = {
        "content": bolt.types.text(),
        "label": bolt.types.categorical(),
    }

    delimiter = "\t"


class CriteoUDTConfig(UDTBenchmarkConfig):
    experiment_name = "CriteoUDT"
    dataset_name = "criteo_46m"
    train_file = "/share/data/udt_datasets/criteo/train_udt.csv"
    test_file = "/share/data/udt_datasets/criteo/test_udt.csv"
    num_epochs = 1

    def get_data_types():
        data_types = {}
        min_vals_of_numeric_cols = np.load(
            "/share/data/udt_datasets/criteo/min_vals_of_numeric_cols.npy"
        )
        max_vals_of_numeric_cols = np.load(
            "/share/data/udt_datasets/criteo/max_vals_of_numeric_cols.npy"
        )

        # Add numerical columns
        for i in range(13):
            data_types[f"num_{i+1}"] = bolt.types.numerical(
                range=(min_vals_of_numeric_cols[i], max_vals_of_numeric_cols[i])
            )

        # Add categorical columns
        for i in range(26):
            data_types[f"cat_{i+1}"] = bolt.types.categorical()

        # Add label column
        data_types["label"] = bolt.types.categorical()
        return data_types


class WayfairUDTConfig(UDTBenchmarkConfig):
    experiment_name = "WayfairUDT"
    dataset_name = "wayfair"
    train_file = "/share/data/wayfair/train_raw_queries.txt"
    test_file = "/share/data/wayfair/dev_raw_queries.txt"
    model_config_path = "wayfair.config"

    data_types = {
        "labels": bolt.types.categorical(delimiter=","),
        "query": bolt.types.text(),
    }
    target = "labels"
    n_target_classes = 931

    # TODO: mlflow does not support parenthetical characters in metric names.
    # We may need to revise our metric naming patterns to use this metric
    # metric_type = "f_measure(0.95)"

    learning_rate = 0.001
    delimiter = "\t"
    # Learning rate scheduler that decreases the learning rate by a factor of 10
    # after the third epoch. This scheduling is what gives us the optimal
    # f-measure on the wayfair dataset.
    callbacks = [
        bolt.callbacks.LearningRateScheduler(
            schedule=bolt.callbacks.MultiStepLR(gamma=0.1, milestones=[3])
        )
    ]

    model_config = {
        "inputs": ["input"],
        "nodes": [
            {
                "name": "hidden",
                "type": "fully_connected",
                "dim": 1024,
                "sparsity": 1.0,
                "activation": "relu",
                "predecessor": "input",
            },
            {
                "name": "output",
                "type": "fully_connected",
                "dim": {"param_name": "output_dim"},
                "sparsity": 0.1,
                "sampling_config": {
                    "num_tables": 64,
                    "hashes_per_table": 4,
                    "reservoir_size": 64,
                },
                "predecessor": "hidden",
            },
        ],
        "output": "output",
        "loss": "CategoricalCrossEntropyLoss",
    }
