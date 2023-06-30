import os
from abc import ABC, abstractmethod

from thirdai import bolt, dataset


class DistributedBenchmarkConfig(ABC):
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


class FiqaConfig(DistributedBenchmarkConfig):
    config_name = "fiqa_config"
    unsupervised_file = "/share/data/mach-doc-search/fiqa/unsupervised.csv"
    supervised_trn = "/share/data/mach-doc-search/fiqa/trn_supervised.csv"
    supervised_tst = "/share/data/mach-doc-search/fiqa/tst_supervised.csv"

    n_target_classes = 57638
    learning_rate = 0.001
    num_epochs = 20
    output_dim = 50000
    num_hashes = 8
    train_metrics = [
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@100",
    ]
    val_metrics = [
        "precision@1",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@100",
    ]


class TreqCovidConfig(DistributedBenchmarkConfig):
    config_name = "trec_covid_config"
    unsupervised_file = "/share/data/mach-doc-search/trec-covid/unsupervised.csv"
    supervised_tst = "/share/data/mach-doc-search/trec-covid/tst_supervised.csv"

    n_target_classes = 171333
    learning_rate = 0.001
    num_epochs = 20
    output_dim = 50000
    num_hashes = 16
    train_metrics = [
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@100",
    ]
    val_metrics = [
        "precision@1",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@100",
    ]
