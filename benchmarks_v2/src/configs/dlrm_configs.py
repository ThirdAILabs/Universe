import os
from abc import ABC, abstractmethod

from thirdai import bolt, dataset


class DLRMConfig(ABC):
    config_name = None
    dataset_name = None

    int_features = None
    cat_features = None
    input_hidden_dim = None
    embedding_args = None
    output_hidden_dim = None
    output_hidden_sparsity = None
    n_classes = None

    learning_rate = None
    num_epochs = None
    delimiter = None
    metrics = ["categorical_accuracy"]

    train_dataset_path = None
    test_dataset_path = None

    @staticmethod
    @abstractmethod
    def get_model():
        pass

    @staticmethod
    @abstractmethod
    def load_datasets(path_prefix: str):
        pass


class CriteoDLRMConfig(DLRMConfig):
    config_name = "bolt_criteo_46m"
    dataset_name = "criteo_46m"

    int_features = 13
    cat_features = 26
    input_hidden_dim = 32
    embedding_args = {
        "num_embedding_lookups": 4,
        "lookup_size": 8,
        "log_embedding_block_size": 20,
        "reduction": "concat",
        "num_tokens_per_input": 26,
    }
    output_hidden_dim = 500
    output_hidden_sparsity = 0.4
    n_classes = 2

    learning_rate = 1e-4
    num_epochs = 1
    delimiter = " "

    train_dataset_path = "criteo/train_shuf.txt"
    test_dataset_path = "criteo/test_shuf.txt"

    def _load_click_through_dataset(
        filename,
        batch_size,
        max_num_numerical_features,
        max_categorical_features,
        delimiter=" ",
    ):
        bolt_dataset, bolt_token_dataset, labels = dataset.load_click_through_dataset(
            filename=filename,
            batch_size=batch_size,
            max_num_numerical_features=max_num_numerical_features,
            max_categorical_features=max_categorical_features,
            delimiter=delimiter,
        )
        return bolt_dataset, bolt_token_dataset, labels

    def load_datasets(path_prefix: str):
        max_num_categorical_features = 26
        num_numerical_features = 13
        batch_size = 512
        (
            train_bolt_dataset,
            train_bolt_token_dataset,
            train_labels,
        ) = CriteoDLRMConfig._load_click_through_dataset(
            filename=os.path.join(path_prefix, CriteoDLRMConfig.train_dataset_path),
            batch_size=batch_size,
            max_categorical_features=max_num_categorical_features,
            max_num_numerical_features=num_numerical_features,
        )
        (
            test_bolt_dataset,
            test_bolt_token_dataset,
            test_labels,
        ) = CriteoDLRMConfig._load_click_through_dataset(
            filename=os.path.join(path_prefix, CriteoDLRMConfig.test_dataset_path),
            batch_size=batch_size,
            max_categorical_features=max_num_categorical_features,
            max_num_numerical_features=num_numerical_features,
        )
        train_data = [train_bolt_dataset, train_bolt_token_dataset]
        test_data = [test_bolt_dataset, test_bolt_token_dataset]

        return train_data, train_labels, test_data, test_labels
