import os
import shutil
from abc import ABC, abstractmethod
from thirdai.demos import download_beir_dataset


def split_into_2(
    file_to_split, destination_file_1, destination_file_2, with_header=False
):
    with open(file_to_split, "r") as input_file:
        with open(destination_file_1, "w+") as f_1:
            with open(destination_file_2, "w+") as f_2:
                for i, line in enumerate(input_file):
                    if with_header and i == 0:
                        f_1.write(line)
                        f_2.write(line)
                        continue

                    if i % 2 == 0:
                        f_1.write(line)
                    else:
                        f_2.write(line)


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
    n_target_classes = None


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

    @classmethod
    def prepare_dataset(cls, path_prefix: str):
        (
            unsupervised_file,
            trn_supervised,
            tst_supervised,
            cls.n_target_classes,
        ) = download_beir_dataset("fiqa")

        directory = os.path.join(path_prefix, cls.dataset_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        split_into_2(
            unsupervised_file,
            os.path.join(path_prefix, cls.unsupervised_file_1),
            os.path.join(path_prefix, cls.unsupervised_file_2),
            with_header=True,
        )
        split_into_2(
            trn_supervised,
            os.path.join(path_prefix, cls.supervised_trn_1),
            os.path.join(path_prefix, cls.supervised_trn_2),
            with_header=True,
        )
        shutil.move(tst_supervised, os.path.join(path_prefix, cls.supervised_tst))


class TreqCovidConfig(DistributedBenchmarkConfig):
    config_name = "trec_covid_config"
    dataset_name = "trec-covid"
    unsupervised_file_1 = "trec-covid/unsupervised_1.csv"
    unsupervised_file_2 = "trec-covid/unsupervised_2.csv"
    supervised_tst = "trec-covid/tst_supervised.csv"

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

    @classmethod
    def prepare_dataset(cls, path_prefix: str):
        (
            unsupervised_file,
            trn_supervised,
            tst_supervised,
            n_target_classes,
        ) = download_beir_dataset("trec-covid")

        directory = os.path.join(path_prefix, cls.dataset_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        split_into_2(
            unsupervised_file,
            os.path.join(path_prefix, cls.unsupervised_file_1),
            os.path.join(path_prefix, cls.unsupervised_file_2),
            with_header=True,
        )
        shutil.move(tst_supervised, os.path.join(path_prefix, cls.supervised_tst))
