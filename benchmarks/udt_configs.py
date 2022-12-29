import numpy as np
from thirdai import bolt


class UDTBenchmarkConfig:
    learning_rate = 0.01
    num_epochs = 5
    target = "label"
    n_target_classes = 2
    delimiter = ","
    metric_type = "categorical_accuracy"
    model_config_path = None


class ClincUDTConfig(UDTBenchmarkConfig):
    train_file = "/share/data/udt_datasets/clinc/clinc_train.csv"
    test_file = "/share/data/udt_datasets/clinc/clinc_test.csv"

    data_types = {
        "text": bolt.types.text(),
        "category": bolt.types.categorical(),
    }

    target = "category"
    n_target_classes = 151
    experiment_name = "ClincUDT"
    dataset_name = "clinc"


class AmazonPolarityUDTConfig(UDTBenchmarkConfig):
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
    experiment_name = "AmazonPolarityUDT"
    dataset_name = "amazon_polarity"


class CriteoUDTConfig(UDTBenchmarkConfig):
    train_file = "/share/data/udt_datasets/criteo/train_udt.csv"
    test_file = "/share/data/udt_datasets/criteo/test_udt.csv"
    num_epochs = 1

    data_types = {}
    min_vals_of_numeric_cols = np.load(
        "/share/data/udt_datasets/criteo/min_vals_of_numeric_cols.npy"
    )
    max_vals_of_numeric_cols = np.load(
        "/share/data/udt_datasets/criteo/max_vals_of_numeric_cols.npy"
    )
    n_unique_classes = np.load("/share/data/udt_datasets/criteo/n_unique_classes.npy")

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

    experiment_name = "CriteoUDT"
    dataset_name = "criteo_46m"


class WayfairUDTConfig(UDTBenchmarkConfig):
    train_file = "/share/data/wayfair/train.txt"
    test_file = "/share/data/wayfair/test.txt"
    model_config_path = "/share/data/wayfair/wayfair_model_config"

    data_types = {
        "labels": bolt.types.categorical(delimiter=","),
        "query": bolt.types.text(),
    }
    target = "labels"
    n_target_classes = 931
    metric_type = "f_measure(0.95)"

    experiment_name = "WayfairUDT"
    dataset_name = "wayfair"
    num_epochs = 1
