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
    experiment_name = "Bolt_Amazon670k"
    rehashing_factor = 6400
    rebuild_hash_tables_factor = 128000

    train_dataset_path = "/share/data/amazon-670k/train_shuffled_noHeader.txt"
    test_dataset_path = "/share/data/amazon-670k/test_shuffled_noHeader_sampled.txt"
    dataset_format = "svm"
    train_batch_size, test_batch_size = 256, 256

    input_dim = 135909
    hidden_dim = 256
    output_dim = 670091
    output_sparsity = 0.005


class Amazon131kConfig(BoltBenchmarkConfig):
    dataset_name = "amazon_131k"
    experiment_name = "Bolt_Amazon131k"
    rehashing_factor = 6400
    rebuild_hash_tables_factor = 128000
    learning_rate = 1e-04

    train_dataset_path = "/share/data/amazon-131k/train_shuffled_noHeader.txt"
    test_dataset_path = "/share/data/amazon-131k/test_shuffled_noHeader_sampled.txt"
    dataset_format = "svm"
    batch_size = 256

    input_dim = 135909
    hidden_dim = 256
    output_dim = 670091
    output_sparsity = 0.005


class AmazonPolarity(BoltBenchmarkConfig):
    dataset_name = "amazon_polarity"
    experiment_name = "Bolt_AmazonPolarity"
    rehashing_factor = 6400
    rebuild_hash_tables_factor = 128000
    learning_rate = 1e-04

    train_dataset_path = "/share/data/amazon_polarity/svm_train.txt"
    test_dataset_path = "/share/data/amazon_polarity/svm_test.txt"
    datasest_format = "svm"
    batch_size = 256

    input_dim = 100000
    hidden_dim = 10000
    hidden_sparsity = 0.005
    output_dim = 2
    output_sparsity = 1.0


class WayfairConfig(BoltBenchmarkConfig):
    dataset_name = "wayfair"
    experiment_name = "Bolt_Wayfair"
    rehashing_factor = 10000
    rebuild_hash_tables_factor = 50000
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
    dataset_name = "criteo_46m"
    experiment_name = "Bolt_Criteo46M"

    rehashing_factor = 6400
    rebuild_hash_tables_factor = 128000
    learning_rate = 1e-04

    train_dataset_path = "/share/data/criteo/train_shuf.txt"
    test_dataset_path = "/share/data/criteo/test_shuf.txt"
    dataset_format = "click_through"
    train_batch_size, test_batch_size = 512, 256
    delimiter = " "
    max_num_numerical_features = 13
    max_num_categorical_features = 26

    nodes = {
        "numerical_input": {"type": "Input", "predecessor": None, "dim": 13},
        "hidden1": {
            "type": "FullyConnected",
            "predecessor": "numerical_input",
            "dim": 1000,
            "sparsity": 0.2,
            "activation": "ReLU",
        },
        "hidden2": {
            "type": "FullyConnected",
            "predecessor": "hidden1",
            "dim": 100,
            "sparsity": 1.0,
            "activation": "ReLU",
        },
        "categorical_input": {
            "type": "Input",
            "predecessor": None,
            "dim": 4294967295,
            "min_num_tokens": 26,
            "max_num_tokens": 26,
        },
        "embedding": {
            "type": "Embedding",
            "predecessor": "categorical_input",
            "num_embedding_lookups": 8,
            "lookup_size": 16,
            "log_embedding_block_size": 20,
            "reduction": "concat",
            "num_tokens_per_input": 26,
        },
        "concat": {"type": "Concatenate", "preds": ["hidden2", "embedding"]},
        "hidden3": {
            "type": "FullyConnected",
            "predecessor": "concat",
            "dim": 1000,
            "sparsity": 0.2,
            "activation": "ReLU",
        },
        "output": {
            "type": "FullyConnected",
            "predecessor": "hidden3",
            "dim": 2,
            "activation": "Softmax",
        },
    }


class FineGrainedBoltBenchmarksConfig(BoltBenchmarkConfig):
    experiment_name = ""

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
