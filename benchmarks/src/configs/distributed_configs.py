import os
import shutil
from abc import ABC, abstractmethod

from thirdai.demos import download_beir_dataset

from ..distributed_utils import split_into_2


class DistributedBenchmarkConfig(ABC):
    config_name = None
    dataset_name = None

    learning_rate = None
    num_epochs = None
    metrics = ["categorical_accuracy"]
    callbacks = []
    n_target_classes = None
    embedding_dimension = 2048

    train_metrics = [
        "precision@1",
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

    @classmethod
    def prepare_dataset(cls, path_prefix: str):
        (
            unsupervised_file,
            trn_supervised,
            tst_supervised,
            cls.n_target_classes,
        ) = download_beir_dataset(cls.dataset_name)

        directory = os.path.join(path_prefix, cls.dataset_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if unsupervised_file is not None:
            split_into_2(
                unsupervised_file,
                os.path.join(path_prefix, cls.unsupervised_file_1),
                os.path.join(path_prefix, cls.unsupervised_file_2),
                with_header=True,
            )
        if trn_supervised is not None:
            split_into_2(
                trn_supervised,
                os.path.join(path_prefix, cls.supervised_trn_1),
                os.path.join(path_prefix, cls.supervised_trn_2),
                with_header=True,
            )
        shutil.move(tst_supervised, os.path.join(path_prefix, cls.supervised_tst))


class FiqaConfig(DistributedBenchmarkConfig):
    config_name = "fiqa_config"
    dataset_name = "fiqa"
    unsupervised_file_1 = "fiqa/unsupervised_1.csv"
    unsupervised_file_2 = "fiqa/unsupervised_2.csv"
    supervised_trn_1 = "fiqa/trn_supervised_1.csv"
    supervised_trn_2 = "fiqa/trn_supervised_2.csv"
    supervised_tst = "fiqa/tst_supervised.csv"

    learning_rate = 0.001
    num_epochs = 20
    output_dim = 50000
    num_hashes = 16


class TreqCovidConfig(DistributedBenchmarkConfig):
    config_name = "trec_covid_config"
    dataset_name = "trec-covid"
    unsupervised_file_1 = "trec-covid/unsupervised_1.csv"
    unsupervised_file_2 = "trec-covid/unsupervised_2.csv"
    supervised_trn_1 = ""  # This dataset doesn't have supervised file
    supervised_trn_2 = ""  # This dataset doesn't have supervised file
    supervised_tst = "trec-covid/tst_supervised.csv"

    learning_rate = 0.001
    num_epochs = 20
    output_dim = 20000
    num_hashes = 16
