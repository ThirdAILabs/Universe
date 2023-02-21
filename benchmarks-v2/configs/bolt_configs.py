from abc import ABC, abstractmethod

from thirdai import bolt, dataset


class BoltBenchmarkConfig(ABC):
    config_name = None
    dataset_name = None

    input_dim = None
    hidden_node = {}
    output_node = {}
    loss_fn = "CategoricalCrossEntropyLoss"
    reconstruct_hash_functions = None
    rebuild_hash_tables = None

    learning_rate = None
    num_epochs = None
    metrics = ["categorical_accuracy"]
    callbacks = []

    @staticmethod
    @abstractmethod
    def load_datasets(path: str):
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
    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000

    learning_rate = 1e-4
    num_epochs = 5

    def load_datasets(path: str):
        train_dataset_path = "amazon-670k/train_shuffled_noHeader.txt"
        test_dataset_path = "amazon-670k/test_shuffled_noHeader_sampled.txt"
        batch_size = 256

        train_data, train_labels = load_svm_dataset(
            filename=path + train_dataset_path, batch_size=batch_size
        )
        test_data, test_labels = load_svm_dataset(
            filename=path + test_dataset_path, batch_size=batch_size
        )
        return train_data, train_labels, test_data, test_labels


class AmazonPolarityConfig(BoltBenchmarkConfig):
    config_name = "bolt_amazon_polarity"
    dataset_name = "amazon_polarity"

    input_dim = 100000
    hidden_node = {"dim": 10000, "sparsity": 0.005, "activation": "ReLU"}
    output_node = {"dim": 2, "activation": "Softmax"}
    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000

    learning_rate = 1e-04
    num_epochs = 5

    def load_datasets(path: str):
        train_dataset_path = "amazon_polarity/svm_train.txt"
        test_dataset_path = "amazon_polarity/svm_test.txt"
        batch_size = 256

        train_data, train_labels = load_svm_dataset(
            filename=path + train_dataset_path, batch_size=batch_size
        )
        test_data, test_labels = load_svm_dataset(
            filename=path + test_dataset_path, batch_size=batch_size
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
            num_tables=64, hashes_per_table=4, reservoir_size=64
        ),
    }
    loss_fn = "BinaryCrossEntropyLoss"
    reconstruct_hash_functions = 10000
    rebuild_hash_tables = 50000

    learning_rate = 1e-04
    num_epochs = 1
    metrics = ["categorical_accuracy", "f_measure(0.95)"]
    callbacks = [
        bolt.callbacks.LearningRateScheduler(
            schedule=bolt.callbacks.MultiStepLR(gamma=0.1, milestones=[3])
        )
    ]

    def _load_wayfair_dataset(filename, batch_size, output_dim, shuffle=True):
        featurizer = dataset.TabularFeaturizer(
            input_blocks=[dataset.blocks.TextPairGram(col=1)],
            label_blocks=[
                dataset.blocks.NumericalId(col=0, n_classes=output_dim, delimiter=",")
            ],
            has_header=False,
            delimiter="\t",
        )

        dataloader = dataset.DatasetLoader(
            data_source=dataset.FileDataSource(filename=filename),
            featurizer=featurizer,
            shuffle=shuffle,
        )
        data, labels = dataloader.load_all(batch_size=batch_size)
        return data, labels

    def load_datasets(path: str):
        train_dataset_path = "wayfair/train_raw_queries.txt"
        test_dataset_path = "wayfair/dev_raw_queries.txt"
        batch_size = 256
        train_data, train_labels = WayfairConfig._load_wayfair_dataset(
            filename=path + train_dataset_path,
            batch_size=batch_size,
            output_dim=WayfairConfig.output_node["dim"],
        )
        test_data, test_labels = WayfairConfig._load_wayfair_dataset(
            filename=path + test_dataset_path,
            batch_size=batch_size,
            output_dim=WayfairConfig.output_node["dim"],
            shuffle=False,
        )
        return train_data, train_labels, test_data, test_labels
