import numpy as np
from thirdai import bolt
from abc import ABC, abstractmethod


class UDTBenchmarkConfig(ABC):
    config_name = None
    dataset_name = None

    train_file = None
    test_file = None

    target = None
    n_target_classes = None
    delimiter = ","
    model_config = None

    learning_rate = None
    num_epochs = None
    callbacks = []
    metrics = ["categorical_accuracy"]

    @staticmethod
    @abstractmethod
    def get_data_types():
        pass


class YelpPolarityUDTConfig(UDTBenchmarkConfig):
    config_name = "yelp_polarity_udt"
    dataset_name = "yelp_polarity"

    train_file = "/share/data/udt_datasets/yelp_polarity/train.csv"
    test_file = "/share/data/udt_datasets/yelp_polarity/test.csv"

    target = "label"
    n_target_classes = 2
    delimiter = "\t"

    learning_rate = 1e-2
    num_epochs = 5

    def get_data_types():
        return {"text": bolt.types.text(), "label": bolt.types.categorical()}


class AmazonPolarityUDTConfig(UDTBenchmarkConfig):
    config_name = "amazon_polarity_udt"
    dataset_name = "amazon_polarity"

    train_file = (
        "/share/data/udt_datasets/amazon_polarity/amazon_polarity_content_train.csv"
    )
    test_file = (
        "/share/data/udt_datasets/amazon_polarity/amazon_polarity_content_test.csv"
    )

    target = "label"
    n_target_classes = 2
    delimiter = "\t"

    learning_rate = 1e-2
    num_epochs = 5

    def get_data_types():
        return {"content": bolt.types.text(), "label": bolt.types.categorical()}


class CriteoUDTConfig(UDTBenchmarkConfig):
    config_name = "criteo_udt"
    dataset_name = "criteo_46m"

    train_file = "/share/data/udt_datasets/criteo/train_udt.csv"
    test_file = "/share/data/udt_datasets/criteo/test_udt.csv"

    target = "label"
    n_target_classes = 2

    learning_rate = 1e-2
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
    config_name = "wayfair_udt"
    dataset_name = "wayfair"

    train_file = "/share/data/wayfair/train_raw_queries.txt"
    test_file = "/share/data/wayfair/dev_raw_queries.txt"

    target = "labels"
    n_target_classes = 931
    delimiter = "\t"
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
                "activation": "sigmoid",
                "sampling_config": {
                    "num_tables": 64,
                    "hashes_per_table": 4,
                    "reservoir_size": 64,
                },
                "predecessor": "hidden",
            },
        ],
        "output": "output",
        "loss": "BinaryCrossEntropyLoss",
    }

    def get_data_types():
        return {
            "labels": bolt.types.categorical(delimiter=","),
            "query": bolt.types.text(),
        }

    learning_rate = 0.001
    num_epochs = 5
    metrics = ["categorical_accuracy", "f_measure(0.95)"]

    # Learning rate scheduler that decreases the learning rate by a factor of 10
    # after the third epoch. This scheduling is what gives us the optimal
    # f-measure on the wayfair dataset.
    callbacks = [
        bolt.callbacks.LearningRateScheduler(
            schedule=bolt.callbacks.MultiStepLR(gamma=0.1, milestones=[3])
        )
    ]
