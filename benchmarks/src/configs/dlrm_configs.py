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
    def load_train_data(path_prefix: str):
        pass

    @staticmethod
    @abstractmethod
    def load_test_data(path_prefix: str):
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

    batch_size = 512

    def _load_click_through_dataset(
        filename,
        batch_size,
        max_num_numerical_features,
        max_categorical_features,
        delimiter=" ",
    ):
        datasets = dataset.load_click_through_dataset(
            filename=filename,
            batch_size=batch_size,
            max_num_numerical_features=max_num_numerical_features,
            max_categorical_features=max_categorical_features,
            delimiter=delimiter,
        )
        return (
            bolt.train.convert_datasets(
                datasets[:2],
                dims=[CriteoDLRMConfig.int_features, 4294967295],
                copy=False,
            ),
            bolt.train.convert_dataset(
                datasets[-1], CriteoDLRMConfig.n_classes, copy=False
            ),
        )

    def load_train_data(path_prefix: str):
        return CriteoDLRMConfig._load_click_through_dataset(
            filename=os.path.join(path_prefix, CriteoDLRMConfig.train_dataset_path),
            batch_size=CriteoDLRMConfig.batch_size,
            max_categorical_features=CriteoDLRMConfig.cat_features,
            max_num_numerical_features=CriteoDLRMConfig.int_features,
        )

    def load_test_data(path_prefix):
        return CriteoDLRMConfig._load_click_through_dataset(
            filename=os.path.join(path_prefix, CriteoDLRMConfig.test_dataset_path),
            batch_size=CriteoDLRMConfig.batch_size,
            max_categorical_features=CriteoDLRMConfig.cat_features,
            max_num_numerical_features=CriteoDLRMConfig.int_features,
        )
