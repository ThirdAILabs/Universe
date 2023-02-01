from configs import BenchmarkConfig
from thirdai import bolt


class BoltBenchmarkConfig(BenchmarkConfig):
    loss_fn = "CategoricalCrossEntropyLoss"
    hidden_sparsity = 1.0
    output_spasity = 1.0
    learning_rate = 1e-04
    hidden_activation = "ReLU"
    hidden_sampling_config = None
    output_activation = "Softmax"
    output_sampling_config = None
    compute_roc_auc = False


# TODO(blaise): Add config for Movie Lens
class Amazon670kConfig(BoltBenchmarkConfig):
    dataset_name = "amazon_670k"
    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000

    train_dataset_path = "amazon-670k/train_shuffled_noHeader.txt"
    test_dataset_path = "amazon-670k/test_shuffled_noHeader_sampled.txt"
    batch_size = 256

    input_dim = 135909
    hidden_dim = 256
    output_dim = 670091
    output_sparsity = 0.005


class Amazon131kConfig(BoltBenchmarkConfig):
    dataset_name = "amazon_131k"
    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000
    learning_rate = 1e-04

    train_dataset_path = "amazon-131k/train_shuffled_noHeader.txt"
    test_dataset_path = "amazon-131k/test_shuffled_noHeader_sampled.txt"
    batch_size = 256

    input_dim = 135909
    hidden_dim = 256
    output_dim = 670091
    output_sparsity = 0.005


class AmazonPolarity(BoltBenchmarkConfig):
    dataset_name = "amazon_polarity"
    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000
    learning_rate = 1e-04

    train_dataset_path = "amazon_polarity/svm_train.txt"
    test_dataset_path = "amazon_polarity/svm_test.txt"
    batch_size = 256

    input_dim = 100000
    hidden_dim = 10000
    hidden_sparsity = 0.005
    output_dim = 2
    output_sparsity = 1.0


class WayfairConfig(BoltBenchmarkConfig):
    dataset_name = "wayfair"
    reconstruct_hash_functions = 10000
    rebuild_hash_tables = 50000
    loss_fn = "CategoricalCrossEntropyLoss"

    train_dataset_path = "/share/data/wayfair/train_raw_queries.txt"
    test_dataset_path = "/share/data/wayfair/dev_raw_queries.txt"
    batch_size = 2048

    input_dim = 100000
    hidden_dim = 1024

    output_dim = 931
    output_activation = "Sigmoid"
    output_sparsity = 0.1
    output_sampling_config = bolt.nn.DWTASamplingConfig(
        num_tables=64, hashes_per_table=4, reservoir_size=64
    )

    callbacks = [
        bolt.callbacks.LearningRateScheduler(
            schedule=bolt.callbacks.MultiStepLR(gamma=0.1, milestones=[3])
        )
    ]


class CriteoDLRMConfig(BoltBenchmarkConfig):
    dataset_name = "criteo"
    reconstruct_hash_functions = 6400
    rebuild_hash_tables = 128000
    learning_rate = 1e-04

    train_dataset_path = "criteo/train_shuf.txt"
    test_dataset_path = "criteo/test_shuf.txt"
    delimiter = " "
    max_num_numerical_features = 13
    max_num_categorical_features = 26

    numerical_input_dim = 13
    first_hidden_dim = 1000
    first_hidden_sparsity = 0.2
    first_hidden_activation = "ReLU"

    second_hidden_dim = 100
    second_hidden_activation = "ReLU"

    categorical_input_dim = 4294967295
    categorical_input_min_num_tokens = 26
    categorical_input_max_num_tokens = 26

    embedding_node_lookups = 8
    embedding_lookup_size = 16
    log_embedding_block_size = 20
    embedding_num_tokens_per_input = 26


class FineGrainedBoltBenchmarks(BoltBenchmarkConfig):
    hidden_and_output_sparsities = [
        0.01,
        0.05,
        0.1,
        0.3,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
