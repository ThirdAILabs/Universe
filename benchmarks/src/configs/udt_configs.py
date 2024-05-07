import os
from abc import ABC, abstractmethod

import pandas as pd
from thirdai import bolt

from .utils import (
    AdditionalMetricCallback,
    get_mae_metric_fn,
    get_mse_metric_fn,
    get_roc_auc_metric_fn,
)


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
    integer_target = False
    callbacks = []
    metrics = ["categorical_accuracy"]
    max_in_memory_batches = None

    # Cold Start configs
    cold_start_learning_rate = None
    cold_start_num_epochs = None
    cold_start_train_file = None
    strong_column_names = []
    weak_column_names = []

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

    learning_rate = 0.01
    num_epochs = 3

    @staticmethod
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

    learning_rate = 0.01
    num_epochs = 3

    @staticmethod
    def get_data_types(path_prefix):
        return {"content": bolt.types.text(), "label": bolt.types.categorical()}


class CriteoUDTConfig(UDTBenchmarkConfig):
    config_name = "criteo_udt"
    dataset_name = "criteo_46m"

    train_file = "udt_datasets/criteo/train_udt.csv"
    test_file = "udt_datasets/criteo/test_udt.csv"

    target = "label"
    n_target_classes = 2

    learning_rate = 0.01
    num_epochs = 1

    max_in_memory_batches = 5000

    @staticmethod
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


class InternetAdsUDTBenchmark(UDTBenchmarkConfig):
    config_name = "internet_ads_udt"
    dataset_name = "internet_ads"

    train_file = "internet_ads/train_with_header.data"
    test_file = "internet_ads/test_with_header.data"

    target = "label"
    n_target_classes = 2

    learning_rate = 0.001
    num_epochs = 100

    metrics = ["categorical_accuracy"]
    callbacks = [
        AdditionalMetricCallback(
            metric_name="roc_auc",
            metric_fn=get_roc_auc_metric_fn(
                target_column="label", positive_label="ad."
            ),
        )
    ]

    @staticmethod
    def get_data_types(path_prefix):
        data_types = {
            "0": bolt.types.numerical(range=(0, 640)),
            "1": bolt.types.numerical(range=(0, 640)),
            "2": bolt.types.numerical(range=(0, 60)),
        }

        for i in range(3, 1558):
            data_types[str(i)] = bolt.types.categorical()

        data_types["label"] = bolt.types.categorical()

        return data_types


class FraudDetectionUDTBenchmark(UDTBenchmarkConfig):
    config_name = "fraud_detection_udt"
    dataset_name = "fraud_detection"

    train_file = "fraud_detection/new_train.csv"
    test_file = "fraud_detection/new_test.csv"

    target = "isFraud"
    n_target_classes = 2

    learning_rate = 0.001
    num_epochs = 2

    metrics = ["categorical_accuracy"]
    callbacks = [
        AdditionalMetricCallback(
            metric_name="roc_auc",
            metric_fn=get_roc_auc_metric_fn(
                target_column="isFraud", positive_label="1"
            ),
        )
    ]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "step": bolt.types.categorical(),
            "type": bolt.types.categorical(),
            "amount": bolt.types.numerical(range=(0, 10000001)),
            "nameOrig": bolt.types.categorical(),
            "oldbalanceOrg": bolt.types.numerical(range=(0, 59585041)),
            "newbalanceOrig": bolt.types.numerical(range=(0, 49585041)),
            "nameDest": bolt.types.categorical(),
            "oldbalanceDest": bolt.types.numerical(range=(0, 356015890)),
            "newbalanceDest": bolt.types.numerical(range=(0, 356179279)),
            "isFraud": bolt.types.categorical(),
            "isFlaggedFraud": bolt.types.categorical(),
        }


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
                "type": "embedding",
                "dim": 1024,
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
                    "range_pow": 12,
                    "binsize": 8,
                    "reservoir_size": 64,
                    "permutations": 8,
                },
                "predecessor": "hidden",
            },
        ],
        "output": "output",
        "loss": "BinaryCrossEntropyLoss",
    }

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "labels": bolt.types.categorical(delimiter=","),
            "query": bolt.types.text(),
        }

    learning_rate = 0.001
    num_epochs = 1
    metrics = ["categorical_accuracy", "f_measure(0.95)"]


class ForestCoverTypeUDTBenchmark(UDTBenchmarkConfig):
    config_name = "forest_cover_type_udt"
    dataset_name = "forest_cover_type"

    train_file = "tabular_benchmarks/ForestCoverType/train_udt_no_index.csv"
    test_file = "tabular_benchmarks/ForestCoverType/test_udt_no_index.csv"

    target = "col54"
    n_target_classes = 7
    delimiter = ","
    options = {"contextual_columns": True}

    learning_rate = 0.01
    num_epochs = 4

    @staticmethod
    def get_data_types(path_prefix):
        return {f"col{i}": bolt.types.categorical() for i in range(55)}


class BlackFridayUDTBenchmark(UDTBenchmarkConfig):
    config_name = "black_friday_udt"
    dataset_name = "black_friday"

    train_file = "tabular_regression/reg_cat/black_friday_train_no_index.csv"
    test_file = "tabular_regression/reg_cat/black_friday_test_no_index.csv"

    target = "Purchase"
    n_target_classes = None
    delimiter = ","
    model_config = None
    options = {"contextual_columns": True}

    learning_rate = 0.001
    num_epochs = 15
    metrics = []
    callbacks = [
        AdditionalMetricCallback(
            metric_name="mse",
            metric_fn=get_mse_metric_fn(target_column="Purchase"),
        ),
        AdditionalMetricCallback(
            metric_name="mae",
            metric_fn=get_mae_metric_fn(target_column="Purchase"),
        ),
    ]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "Gender": bolt.types.categorical(),
            "Age": bolt.types.categorical(),
            "Occupation": bolt.types.categorical(),
            "City_Category": bolt.types.categorical(),
            "Stay_In_Current_City_Years": bolt.types.categorical(),
            "Marital_Status": bolt.types.categorical(),
            "Product_Category_1": bolt.types.categorical(),
            "Product_Category_2": bolt.types.categorical(),
            "Product_Category_3": bolt.types.categorical(),
            "Purchase": bolt.types.numerical(range=(5, 11)),
        }


class DiamondsUDTBenchmark(UDTBenchmarkConfig):
    config_name = "diamonds_udt"
    dataset_name = "diamonds"

    train_file = "tabular_regression/reg_cat/diamonds_train_no_index.csv"
    test_file = "tabular_regression/reg_cat/diamonds_test_no_index.csv"

    target = "price"
    n_target_classes = None
    delimiter = ","

    learning_rate = 0.001
    num_epochs = 15
    metrics = []
    callbacks = [
        AdditionalMetricCallback(
            metric_name="mse",
            metric_fn=get_mse_metric_fn(target_column="price"),
        ),
        AdditionalMetricCallback(
            metric_name="mae",
            metric_fn=get_mae_metric_fn(target_column="price"),
        ),
    ]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "carat": bolt.types.numerical(range=(0.2, 5.01)),
            "cut": bolt.types.categorical(),
            "color": bolt.types.categorical(),
            "clarity": bolt.types.categorical(),
            "depth": bolt.types.numerical(range=(43, 79)),
            "table": bolt.types.numerical(range=(43, 79)),
            "x": bolt.types.numerical(range=(0, 10.74)),
            "y": bolt.types.numerical(range=(0, 58.9)),
            "z": bolt.types.numerical(range=(0, 31.8)),
            "price": bolt.types.numerical(
                range=(5.7899601708972535, 9.842887930407198)
            ),
        }


class MercedesBenzGreenerUDTBenchmark(UDTBenchmarkConfig):
    config_name = "mercedes_benz_greener_udt"
    dataset_name = "mercedes_benz_greener"

    train_file = "tabular_regression/reg_cat/Mercedes_Benz_Greener_Manufacturing_train_no_index.csv"
    test_file = "tabular_regression/reg_cat/Mercedes_Benz_Greener_Manufacturing_test_no_index.csv"

    target = "y"
    n_target_classes = None
    delimiter = ","

    learning_rate = 0.001
    num_epochs = 15
    metrics = []
    callbacks = [
        AdditionalMetricCallback(
            metric_name="mse",
            metric_fn=get_mse_metric_fn(target_column="y"),
        ),
        AdditionalMetricCallback(
            metric_name="mae",
            metric_fn=get_mae_metric_fn(target_column="y"),
        ),
    ]

    @staticmethod
    def get_data_types(path_prefix):
        filename = os.path.join(path_prefix, MercedesBenzGreenerUDTBenchmark.train_file)
        with open(filename) as f:
            column_names = f.readline().strip().split(",")

        data_types = {
            f"X{i}": bolt.types.categorical()
            for i in range(3, 386)
            if f"X{i}" in column_names
        }

        data_types["y"] = bolt.types.numerical(range=(72.5, 265.32))
        return data_types


class TranslitUDTBenchmark(UDTBenchmarkConfig):
    config_name = "translit_udt"
    dataset_name = "translit"

    train_file = "lstm_translit/train_asm.csv"
    test_file = "lstm_translit/test_asm.csv"

    target = "output_seq"
    n_target_classes = 26
    delimiter = ","

    num_epochs = 5
    learning_rate = 0.001
    metrics = ["categorical_accuracy"]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "input_seq": bolt.types.sequence(delimiter=" "),
            "output_seq": bolt.types.sequence(max_length=30, delimiter=" "),
        }
