import numpy as np

from thirdai import bolt


class UDTBenchmarkConfig():
    learning_rate = 0.01
    num_epochs = 5
    target = "label"
    n_target_classes = 2
    delimiter = ","


class ClincConfig(UDTBenchmarkConfig):
    train_file = "/share/data/udt_datasets/clinc/clinc_train.csv"
    test_file = "/share/data/udt_datasets/clinc/clinc_test.csv"

    data_types = {
        "text": bolt.types.text(),
        "category": bolt.types.categorical(),
    }

    target = "category"
    n_target_classes = 151


class AmazonPolarityConfig(UDTBenchmarkConfig):
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


class CriteoConfig(UDTBenchmarkConfig):
    train_file = "/share/data/udt_datasets/criteo/train_udt.csv"
    test_file = "/share/data/udt_datasets/criteo/test_udt.csv"

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

