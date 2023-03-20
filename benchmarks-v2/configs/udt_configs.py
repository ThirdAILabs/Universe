import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from thirdai import bolt, data


class UDTBenchmarkConfig(ABC):
    config_name = None
    dataset_name = None

    train_file = None
    test_file = None

    target = None
    n_target_classes = None
    temporal_relationships = {}
    delimiter = ","
    model_config = None
    options = {}

    learning_rate = None
    num_epochs = None
    callbacks = []
    metrics = ["categorical_accuracy"]
    additional_metric_fn = None

    @staticmethod
    @abstractmethod
    def get_data_types(path_prefix):
        pass


class YelpPolarityUDTConfig(UDTBenchmarkConfig):
    config_name = "yelp_polarity_udt"
    dataset_name = "yelp_polarity"

    train_file = "udt_datasets/yelp_polarity/train.csv"
    test_file = "udt_datasets/yelp_polarity/test.csv"

    target = "label"
    n_target_classes = 2
    delimiter = "\t"

    learning_rate = 1e-2
    num_epochs = 3

    def get_data_types(path_prefix):
        return {"text": bolt.types.text(), "label": bolt.types.categorical()}


class AmazonPolarityUDTConfig(UDTBenchmarkConfig):
    config_name = "amazon_polarity_udt"
    dataset_name = "amazon_polarity"

    train_file = "udt_datasets/amazon_polarity/amazon_polarity_content_train.csv"
    test_file = "udt_datasets/amazon_polarity/amazon_polarity_content_test.csv"

    target = "label"
    n_target_classes = 2
    delimiter = "\t"

    learning_rate = 1e-2
    num_epochs = 3

    def get_data_types(path_prefix):
        return {"content": bolt.types.text(), "label": bolt.types.categorical()}


class CriteoUDTConfig(UDTBenchmarkConfig):
    config_name = "criteo_udt"
    dataset_name = "criteo_46m"

    train_file = "udt_datasets/criteo/train_udt.csv"
    test_file = "udt_datasets/criteo/test_udt.csv"

    target = "label"
    n_target_classes = 2

    learning_rate = 1e-2
    num_epochs = 1

    def get_data_types(path_prefix):
        data_types = {}

        numeric_col_ranges = [
            (0.0, 8.66),
            (0.0, 12.46),
            (0.0, 11.09),
            (0.0, 6.88),
            (0.0, 16.96),
            (0.0, 12.97),
            (0.0, 10.94),
            (0.0, 8.71),
            (0.0, 10.28),
            (0.0, 2.48),
            (0.0, 5.45),
            (0.0, 8.3),
            (0.0, 8.91),
        ]

        # Add numerical columns
        for i, min_max in enumerate(numeric_col_ranges):
            data_types[f"num_{i+1}"] = bolt.types.numerical(range=min_max)

        # Add categorical columns
        for i in range(26):
            data_types[f"cat_{i+1}"] = bolt.types.categorical()

        # Add label column
        data_types["label"] = bolt.types.categorical()
        return data_types


class WayfairUDTConfig(UDTBenchmarkConfig):
    config_name = "wayfair_udt"
    dataset_name = "wayfair"

    train_file = "wayfair/train_raw_queries.txt"
    test_file = "wayfair/dev_raw_queries.txt"

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

    def get_data_types(path_prefix):
        return {
            "labels": bolt.types.categorical(delimiter=","),
            "query": bolt.types.text(),
        }

    learning_rate = 0.001
    num_epochs = 1
    metrics = ["categorical_accuracy", "f_measure(0.95)"]

    # Learning rate scheduler that decreases the learning rate by a factor of 10
    # after the third epoch. This scheduling is what gives us the optimal
    # f-measure on the wayfair dataset.
    callbacks = [
        bolt.callbacks.LearningRateScheduler(
            schedule=bolt.callbacks.MultiStepLR(gamma=0.1, milestones=[3])
        )
    ]


class MovieLensUDTBenchmark(UDTBenchmarkConfig):
    config_name = "movie_lens_udt"
    dataset_name = "movie_lens"

    train_file = "movielens1m/train.csv"
    test_file = "movielens1m/test.csv"

    target = "movieId"
    n_target_classes = 3706
    temporal_relationships = {
        "userId": [
            bolt.temporal.categorical(column_name="movieId", track_last_n=length)
            for length in [1, 2, 5, 10, 25, 50]
        ]
    }

    learning_rate = 0.0001
    num_epochs = 5
    metrics = ["recall@10"]

    @staticmethod
    @abstractmethod
    def get_data_types(path_prefix):
        return {
            "userId": bolt.types.categorical(),
            "movieId": bolt.types.categorical(delimiter=" "),
            "timestamp": bolt.types.date(),
        }


class ForestCoverTypeUDTBenchmark(UDTBenchmarkConfig):
    config_name = "forest_cover_type_udt"
    dataset_name = "forest_cover_type"

    train_file = "tabular_benchmarks/ForestCoverType/train_udt.csv"
    test_file = "tabular_benchmarks/ForestCoverType/test_udt.csv"

    target = "col54"
    n_target_classes = 7
    delimiter = ","
    options = {"contextual_columns": True}

    learning_rate = 0.01
    num_epochs = 4

    @staticmethod
    @abstractmethod
    def get_data_types(path_prefix):
        return {f"col{i}": bolt.types.categorical() for i in range(55)}


def regression_metrics(activations, test_file, mlflow_logger, step):
    df = pd.read_csv(test_file)
    labels = df["Purchase"].to_numpy()

    mae = np.mean(np.abs(activations - labels))
    mse = np.mean(np.square(activations - labels))

    print(f"MAE = {mae}\nMSE = {mse}")
    if mlflow_logger:
        mlflow_logger.log_additional_metric(key="mae", value=mae, step=step)
        mlflow_logger.log_additional_metric(key="mse", value=mse, step=step)


class BlackFridayUDTBenchmark(UDTBenchmarkConfig):
    config_name = "black_friday_udt"
    dataset_name = "black_friday"

    train_file = "tabular_regression/reg_cat/black_friday_shuf_train_with_header.csv"
    test_file = "tabular_regression/reg_cat/black_friday_test.csv"

    target = "Purchase"
    n_target_classes = None
    delimiter = ","
    model_config = None
    options = {"contextual_columns": True}

    learning_rate = 0.001
    num_epochs = 15
    metrics = []
    additional_metric_fn = regression_metrics

    @staticmethod
    @abstractmethod
    def get_data_types(path_prefix):
        file = os.path.join(path_prefix, BlackFridayUDTBenchmark.train_file)

        col_types = data.get_udt_col_types(file)
        del col_types["Unnamed: 0"]
        return col_types
