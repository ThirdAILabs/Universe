import os
from abc import ABC, abstractmethod

from thirdai import bolt, data

from .utils import (
    AdditionalMetricCallback,
    gnn_roc_auc_with_target_name,
    mach_precision_at_1_with_target_name,
    mach_recall_at_5_with_target_name,
    mae_with_target_name,
    mse_with_target_name,
    pokec_col_ranges,
    roc_auc_with_target_name,
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
    model_config_path = None
    options = {}

    learning_rate = None
    num_epochs = None
    integer_target = False
    callbacks = []
    metrics = ["categorical_accuracy"]

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

    learning_rate = 1e-2
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

    learning_rate = 1e-2
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

    learning_rate = 1e-2
    num_epochs = 1

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

    learning_rate = 1e-3
    num_epochs = 100

    metrics = ["categorical_accuracy"]
    callbacks = [
        AdditionalMetricCallback(
            metric_name="roc_auc",
            metric_fn=roc_auc_with_target_name("label", "ad."),
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

    learning_rate = 1e-3
    num_epochs = 2

    metrics = ["categorical_accuracy"]
    callbacks = [
        AdditionalMetricCallback(
            metric_name="roc_auc",
            metric_fn=roc_auc_with_target_name("isFraud", "1"),
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

    train_file = "tabular_benchmarks/ForestCoverType/train_udt.csv"
    test_file = "tabular_benchmarks/ForestCoverType/test_udt.csv"

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
    callbacks = [
        AdditionalMetricCallback(
            metric_name="mse",
            metric_fn=mse_with_target_name("Purchase"),
        ),
        AdditionalMetricCallback(
            metric_name="mae",
            metric_fn=mae_with_target_name("Purchase"),
        ),
    ]

    @staticmethod
    def get_data_types(path_prefix):
        file = os.path.join(path_prefix, BlackFridayUDTBenchmark.train_file)

        col_types = data.get_udt_col_types(file)
        del col_types["Unnamed: 0"]
        return col_types


class DiamondsUDTBenchmark(UDTBenchmarkConfig):
    config_name = "diamonds_udt"
    dataset_name = "diamonds"

    train_file = "tabular_regression/reg_cat/diamonds_shuf_train_with_header.csv"
    test_file = "tabular_regression/reg_cat/diamonds_test.csv"

    target = "price"
    n_target_classes = None
    delimiter = ","

    learning_rate = 0.001
    num_epochs = 15
    metrics = []
    callbacks = [
        AdditionalMetricCallback(
            metric_name="mse",
            metric_fn=mse_with_target_name("price"),
        ),
        AdditionalMetricCallback(
            metric_name="mae",
            metric_fn=mae_with_target_name("price"),
        ),
    ]

    @staticmethod
    def get_data_types(path_prefix):
        file = os.path.join(path_prefix, DiamondsUDTBenchmark.train_file)

        col_types = data.get_udt_col_types(file)
        del col_types["Unnamed: 0"]
        return col_types


class MercedesBenzGreenerUDTBenchmark(UDTBenchmarkConfig):
    config_name = "mercedes_benz_greener_udt"
    dataset_name = "mercedes_benz_greener"

    train_file = "tabular_regression/reg_cat/Mercedes_Benz_Greener_Manufacturing_shuf_train_with_header.csv"
    test_file = (
        "tabular_regression/reg_cat/Mercedes_Benz_Greener_Manufacturing_test.csv"
    )

    target = "y"
    n_target_classes = None
    delimiter = ","

    learning_rate = 0.001
    num_epochs = 15
    metrics = []
    callbacks = [
        AdditionalMetricCallback(
            metric_name="mse",
            metric_fn=mse_with_target_name("y"),
        ),
        AdditionalMetricCallback(
            metric_name="mae",
            metric_fn=mae_with_target_name("y"),
        ),
    ]

    @staticmethod
    def get_data_types(path_prefix):
        file = os.path.join(path_prefix, MercedesBenzGreenerUDTBenchmark.train_file)

        col_types = data.get_udt_col_types(file)
        del col_types["Unnamed: 0"]
        return col_types


class ScifactColdStartUDTBenchmark(UDTBenchmarkConfig):
    config_name = "scifact_cold_start_udt"
    dataset_name = "scifact"

    cold_start_train_file = "scifact/unsupervised.csv"
    test_file = "scifact/tst_supervised.csv"

    target = "DOC_ID"
    integer_target = True
    n_target_classes = 5183
    delimiter = ","

    metrics = ["precision@1", "recall@5"]
    cold_start_num_epochs = 5
    cold_start_learning_rate = 1e-3
    strong_column_names = ["TITLE"]
    weak_column_names = ["TEXT"]
    options = {"embedding_dimension": 1024}

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        }


class CookingColdStartUDTBenchmark(UDTBenchmarkConfig):
    config_name = "cooking_cold_start_udt"
    dataset_name = "cooking"

    cold_start_train_file = (
        "catalog_recommender/cooking/reformatted_trn_unsupervised.csv"
    )
    test_file = "catalog_recommender/cooking/reformatted_tst_supervised.csv"

    target = "LABEL_IDS"
    integer_target = True
    n_target_classes = 26109
    delimiter = ","
    model_config_path = "catalog_recommender/cooking/cooking_model_config"

    metrics = ["precision@1", "recall@100"]
    cold_start_num_epochs = 15
    cold_start_learning_rate = 1e-3
    strong_column_names = []
    weak_column_names = ["DESCRIPTION", "BRAND"]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "LABEL_IDS": bolt.types.categorical(delimiter=";"),
            "QUERY": bolt.types.text(contextual_encoding="local"),
        }


class ScifactMachUDTBenchmark(UDTBenchmarkConfig):
    config_name = "scifact_mach_udt"
    dataset_name = "scifact"

    train_file = "scifact/trn_supervised.csv"
    test_file = "scifact/tst_supervised.csv"

    target = "DOC_ID"
    integer_target = True
    n_target_classes = 5183
    delimiter = ","

    num_epochs = 10
    learning_rate = 1e-3
    options = {"extreme_classification": True, "embedding_dimension": 1024}
    metrics = []
    callbacks = [
        AdditionalMetricCallback(
            metric_name="precision@1",
            metric_fn=mach_precision_at_1_with_target_name(
                "DOC_ID", target_delimeter=":"
            ),
        ),
        AdditionalMetricCallback(
            metric_name="recall@5",
            metric_fn=mach_recall_at_5_with_target_name("DOC_ID", target_delimeter=":"),
        ),
    ]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        }


class YelpChiUDTBenchmark(UDTBenchmarkConfig):
    config_name = "yelp_chi_udt"
    dataset_name = "yelp_chi"

    train_file = "yelp_chi/yelp_train.csv"
    test_file = "yelp_chi/yelp_test.csv"

    target = "target"
    integer_target = True
    n_target_classes = 2
    delimiter = ","

    num_epochs = 15
    learning_rate = 1e-3
    metrics = ["categorical_accuracy"]
    callbacks = [
        AdditionalMetricCallback(
            metric_name="roc_auc",
            metric_fn=gnn_roc_auc_with_target_name("target"),
        )
    ]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "node_id": bolt.types.node_id(),
            **{f"col_{i}": bolt.types.numerical([0, 1]) for i in range(32)},
            "target": bolt.types.categorical(),
            "neighbors": bolt.types.neighbors(),
        }


class PokecUDTBenchmark(UDTBenchmarkConfig):
    config_name = "pokec_udt"
    dataset_name = "pokec"

    train_file = "pokec/train.csv"
    test_file = "pokec/train.csv"

    target = "target"
    integer_target = True
    n_target_classes = 2
    delimiter = ","

    num_epochs = 15
    learning_rate = 1e-3
    metrics = ["categorical_accuracy"]
    callbacks = [
        AdditionalMetricCallback(
            metric_name="roc_auc",
            metric_fn=gnn_roc_auc_with_target_name("target"),
        )
    ]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "node_id": bolt.types.node_id(),
            **{
                f"col_{col_name}": bolt.types.numerical(col_range)
                for col_name, col_range in enumerate(pokec_col_ranges)
            },
            "target": bolt.types.categorical(),
            "neighbors": bolt.types.neighbors(),
        }


class TranslitUDTBenchmark(UDTBenchmarkConfig):
    config_name = "translit_udt"
    dataset_name = "translit"

    train_file = "lstm_translit/train_asm.csv"
    test_file = "lstm_translit/test_asm.csv"

    target = "output_seq"
    n_target_classes = 26
    delimiter = ","

    num_epochs = 5
    learning_rate = 1e-3
    metrics = ["categorical_accuracy"]

    @staticmethod
    def get_data_types(path_prefix):
        return {
            "input_seq": bolt.types.sequence(delimiter=" "),
            "output_seq": bolt.types.sequence(max_length=30, delimiter=" "),
        }
