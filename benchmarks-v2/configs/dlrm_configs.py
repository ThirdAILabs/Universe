from configs.base_configs import DLRMConfig
from thirdai import dataset


class CriteoDLRMConfig(DLRMConfig):
    dataset_name = "criteo_46m"
    experiment_name = "Bolt_Criteo46M"

    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000
    num_epochs = 5
    learning_rate = 1e-04

    delimiter = " "
    train_dataset_path = "/share/data/criteo/train_shuf.txt"
    test_dataset_path = "/share/data/criteo/test_shuf.txt"
    train_batch_size, test_batch_size = 512, 256
    max_num_numerical_features = 13
    max_num_categorical_features = 26

    compute_roc_auc = True
    input_dim = 13
    token_input = {"dim": 4294967295, "num_tokens_range": (26, 26)}
    first_hidden_node = {"dim": 1000, "sparsity": 0.2, "activation": "ReLU"}
    second_hidden_node = {"dim": 100, "activation": "ReLU"}
    embedding_node = {
        "num_embedding_lookups": 8,
        "lookup_size": 16,
        "log_embedding_block_size": 20,
        "reduction": "concat",
        "num_tokens_per_input": 26,
    }
    third_hidden_node = {
        "dim": 1000,
        "sparsity": 0.2,
        "activation": "ReLU",
    }
    output_node = {
        "dim": 2,
        "activation": "Softmax",
    }

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

    def load_datasets():
        (
            train_bolt_dataset,
            train_bolt_token_dataset,
            train_labels,
        ) = CriteoDLRMConfig._load_click_through_dataset(
            filename=CriteoDLRMConfig.train_dataset_path,
            batch_size=CriteoDLRMConfig.train_batch_size,
            max_categorical_features=CriteoDLRMConfig.max_num_categorical_features,
            max_num_numerical_features=CriteoDLRMConfig.max_num_numerical_features,
        )
        (
            test_bolt_dataset,
            test_bolt_token_dataset,
            test_labels,
        ) = CriteoDLRMConfig._load_click_through_dataset(
            filename=CriteoDLRMConfig.test_dataset_path,
            batch_size=CriteoDLRMConfig.test_batch_size,
            max_categorical_features=CriteoDLRMConfig.max_num_categorical_features,
            max_num_numerical_features=CriteoDLRMConfig.max_num_numerical_features,
        )
        train_data = [train_bolt_dataset, train_bolt_token_dataset]
        test_data = [test_bolt_dataset, test_bolt_token_dataset]

        return train_data, train_labels, test_data, test_labels
