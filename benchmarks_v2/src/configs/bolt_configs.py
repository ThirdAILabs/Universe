import os
from abc import ABC, abstractmethod

from thirdai import bolt, dataset


class BoltBenchmarkConfig(ABC):
    config_name = None
    dataset_name = None

    input_dim = None
    hidden_node = {}
    output_node = {}
    loss_fn = "CategoricalCrossEntropyLoss"
    rebuild_hash_tables = None
    reconstruct_hash_functions = None

    learning_rate = None
    num_epochs = None
    metrics = ["categorical_accuracy"]
    callbacks = []

    @staticmethod
    @abstractmethod
    def load_datasets(path_prefix: str):
        pass


def load_svm_dataset(filename, batch_size):
    data, labels = dataset.load_bolt_svm_dataset(
        filename=filename, batch_size=batch_size
    )
    return data, labels


class Amazon670kConfig(BoltBenchmarkConfig):
    config_name = "bolt_amazon_670k"
    dataset_name = "amazon_670k"

    input_dim = 135909
    hidden_node = {"dim": 256, "activation": "ReLU"}
    output_node = {"dim": 670091, "sparsity": 0.005, "activation": "Softmax"}
    rebuild_hash_tables = 6400
    reconstruct_hash_functions = 128000

    batch_size = 256
    learning_rate = 1e-4
    num_epochs = 5

    def load_datasets(path_prefix: str):
        train_dataset_path = os.path.join(
            path_prefix, "amazon-670k/train_shuffled_noHeader.txt"
        )
        test_dataset_path = os.path.join(
            path_prefix, "amazon-670k/test_shuffled_noHeader_sampled.txt"
        )

        train_data, train_labels = load_svm_dataset(
            filename=train_dataset_path, batch_size=Amazon670kConfig.batch_size
        )
        test_data, test_labels = load_svm_dataset(
            filename=test_dataset_path, batch_size=Amazon670kConfig.batch_size
        )
        return train_data, train_labels, test_data, test_labels


class AmazonPolarityConfig(BoltBenchmarkConfig):
    config_name = "bolt_amazon_polarity"
    dataset_name = "amazon_polarity"

    input_dim = 100000
    hidden_node = {"dim": 10000, "sparsity": 0.005, "activation": "ReLU"}
    output_node = {"dim": 2, "activation": "Softmax"}
    rebuild_hash_tables = 6400
    reconstruct_hash_functions = 128000

    batch_size = 256
    learning_rate = 1e-04
    num_epochs = 5

    def load_datasets(path_prefix: str):
        train_dataset_path = os.path.join(path_prefix, "amazon_polarity/svm_train.txt")
        test_dataset_path = os.path.join(path_prefix, "amazon_polarity/svm_test.txt")

        train_data, train_labels = load_svm_dataset(
            filename=train_dataset_path, batch_size=AmazonPolarityConfig.batch_size
        )
        test_data, test_labels = load_svm_dataset(
            filename=test_dataset_path, batch_size=AmazonPolarityConfig.batch_size
        )
        return train_data, train_labels, test_data, test_labels


class WayfairConfig(BoltBenchmarkConfig):
    config_name = "bolt_wayfair"
    dataset_name = "wayfair"

    input_dim = 100000
    hidden_node = {"dim": 1024, "activation": "ReLU"}
    output_node = {
        "dim": 931,
        "sparsity": 0.1,
        "activation": "Sigmoid",
        "sampling_config": bolt.nn.DWTASamplingConfig(
            num_tables=64,
            hashes_per_table=4,
            range_pow=12,
            binsize=8,
            reservoir_size=64,
            permutations=8,
        ),
    }
    loss_fn = "BinaryCrossEntropyLoss"
    rebuild_hash_tables = 10000
    reconstruct_hash_functions = 50000

    batch_size = 2048
    learning_rate = 1e-04
    num_epochs = 5
    metrics = ["categorical_accuracy", "f_measure(0.95)"]

    def _load_wayfair_dataset(filename, batch_size, output_dim, shuffle=True):
        featurizer = dataset.TabularFeaturizer(
            block_lists=[
                dataset.BlockList(
                    [dataset.blocks.TextBlock(col=1, encoder=dataset.PairGramEncoder())]
                ),
                dataset.BlockList(
                    [
                        dataset.blocks.NumericalId(
                            col=0, n_classes=output_dim, delimiter=","
                        )
                    ]
                ),
            ],
            has_header=True,
            delimiter="\t",
        )

        dataloader = dataset.DatasetLoader(
            data_source=dataset.FileDataSource(filename=filename),
            featurizer=featurizer,
            shuffle=shuffle,
        )
        data, labels = dataloader.load_all(batch_size=batch_size)
        return data, labels

    def load_datasets(path_prefix: str):
        train_dataset_path = os.path.join(path_prefix, "wayfair/train_raw_queries.txt")
        test_dataset_path = os.path.join(path_prefix, "wayfair/dev_raw_queries.txt")
        train_data, train_labels = WayfairConfig._load_wayfair_dataset(
            filename=train_dataset_path,
            batch_size=WayfairConfig.batch_size,
            output_dim=WayfairConfig.output_node["dim"],
        )
        test_data, test_labels = WayfairConfig._load_wayfair_dataset(
            filename=test_dataset_path,
            batch_size=WayfairConfig.batch_size,
            output_dim=WayfairConfig.output_node["dim"],
            shuffle=False,
        )
        return train_data, train_labels, test_data, test_labels
