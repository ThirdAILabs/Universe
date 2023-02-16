from configs.base_configs import BoltBenchmarkConfig, DLRMConfig
from thirdai import bolt, dataset


def load_svm_dataset(filename, batch_size):
    data, labels = dataset.load_bolt_svm_dataset(
        filename=filename,
        batch_size=batch_size,
    )
    return data, labels


class MovieLensConfig(BoltBenchmarkConfig):
    dataset_name = "movielens1m"
    experiment_name = "Bolt_MovieLens1M"
    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000
    num_epochs = 5
    learning_rate = 1e-04



class Amazon670kConfig(BoltBenchmarkConfig):
    dataset_name = "amazon_670k"
    experiment_name = "Bolt_Amazon670k"
    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000
    num_epochs = 1
    learning_rate = 1e-04

    train_dataset_path = "/share/data/amazon-670k/train_shuffled_noHeader.txt"
    test_dataset_path = "/share/data/amazon-670k/test_shuffled_noHeader_sampled.txt"
    train_batch_size, test_batch_size = 256, 256

    input_dim = 135909
    hidden_node = {"dim": 256, "activation": "ReLU"}
    output_node = {"dim": 670091, "sparsity": 0.005, "activation": "Softmax"}

    def load_datasets():
        train_data, train_labels = load_svm_dataset(
            filename=Amazon670kConfig.train_dataset_path,
            batch_size=Amazon670kConfig.train_batch_size,
        )
        test_data, test_labels = load_svm_dataset(
            filename=Amazon670kConfig.test_dataset_path,
            batch_size=Amazon670kConfig.test_batch_size,
        )
        return train_data, train_labels, test_data, test_labels


class Amazon131kConfig(BoltBenchmarkConfig):
    dataset_name = "amazon_131k"
    experiment_name = "Bolt_Amazon131k"
    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000
    num_epochs = 1
    learning_rate = 1e-04

    train_dataset_path = "/share/data/amazon-131k/train_shuffled_noHeader.txt"
    test_dataset_path = "/share/data/amazon-131k/test_shuffled_noHeader_sampled.txt"
    train_batch_size, test_batch_size = 256, 256

    input_dim = 135909
    hidden_node = {"dim": 256, "activation": "ReLU"}
    output_node = {"dim": 670091, "sparsity": 0.005, "activation": "Softmax"}

    def load_datasets():
        train_data, train_labels = load_svm_dataset(
            filename=Amazon131kConfig.train_dataset_path,
            batch_size=Amazon131kConfig.train_batch_size,
        )
        test_data, test_labels = load_svm_dataset(
            filename=Amazon131kConfig.test_dataset_path,
            batch_size=Amazon131kConfig.test_batch_size,
        )
        return train_data, train_labels, test_data, test_labels


class AmazonPolarityConfig(BoltBenchmarkConfig):
    dataset_name = "amazon_polarity"
    experiment_name = "Bolt_AmazonPolarity"
    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000
    num_epochs = 1
    learning_rate = 1e-04

    train_dataset_path = "/share/data/amazon_polarity/svm_train.txt"
    test_dataset_path = "/share/data/amazon_polarity/svm_test.txt"
    train_batch_size, test_batch_size = 256, 256

    input_dim = 100000
    hidden_node = {"dim": 10000, "sparsity": 0.005, "activation": "ReLU"}
    output_node = {"dim": 2, "activation": "Softmax"}

    def load_datasets():
        train_data, train_labels = load_svm_dataset(
            filename=AmazonPolarityConfig.train_dataset_path,
            batch_size=AmazonPolarityConfig.train_batch_size,
        )
        test_data, test_labels = load_svm_dataset(
            filename=AmazonPolarityConfig.test_dataset_path,
            batch_size=AmazonPolarityConfig.test_batch_size,
        )
        return train_data, train_labels, test_data, test_labels


class WayfairConfig(BoltBenchmarkConfig):
    dataset_name = "wayfair"
    experiment_name = "Bolt_Wayfair"
    reconstruct_hash_functions = 10000
    rebuild_hash_tables = 50000
    num_epochs = 1
    learning_rate = 1e-04

    train_dataset_path = "/share/data/wayfair/train_raw_queries.txt"
    test_dataset_path = "/share/data/wayfair/dev_raw_queries.txt"
    train_batch_size, test_batch_size = 2048, 2048

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

    callbacks = [
        bolt.callbacks.LearningRateScheduler(
            schedule=bolt.callbacks.MultiStepLR(gamma=0.1, milestones=[3])
        )
    ]

    def _load_wayfair_dataset(filename, batch_size, output_dim, shuffle=True):
        batch_processor = dataset.GenericBatchProcessor(
            input_blocks=[dataset.blocks.TextPairGram(col=1)],
            label_blocks=[
                dataset.blocks.NumericalId(col=0, n_classes=output_dim, delimiter=",")
            ],
            has_header=False,
            delimiter="\t",
        )

        dataloader = dataset.DatasetLoader(
            data_source=dataset.FileDataSource(
                filename=filename, batch_size=batch_size
            ),
            batch_processor=batch_processor,
            shuffle=shuffle,
        )
        data, labels = dataloader.load_in_memory()
        return data, labels

    def load_datasets():
        train_data, train_labels = WayfairConfig._load_wayfair_dataset(
            filename=WayfairConfig.train_dataset_path,
            batch_size=WayfairConfig.train_batch_size,
            output_dim=WayfairConfig.output_dim,
        )
        test_data, test_labels = WayfairConfig._load_wayfair_dataset(
            filename=WayfairConfig.test_dataset_path,
            batch_size=WayfairConfig.test_batch_size,
            output_dim=WayfairConfig.output_dim,
            shuffle=False,
        )
        return train_data, train_labels, test_data, test_labels
