import os

import datasets
import pytest
from utils import (
    ray_two_node_cluster_config,
    write_dataset_to_csv_clinc,
    remove_files,
)

from thirdai import bolt, data

pytestmark = [pytest.mark.distributed]

TRAIN_FILE_1 = "./clinc_train_distributed_1.csv"
TRAIN_FILE_2 = "./clinc_train_distributed_2.csv"
TEST_FILE = "./clinc_test_distributed.csv"


def download_clinc_dataset():
    clinc_dataset = datasets.load_dataset("clinc_oos", "small")
    write_dataset_to_csv_clinc(clinc_dataset["train"], TRAIN_FILE_1, 0, 2)
    write_dataset_to_csv_clinc(clinc_dataset["test"], TRAIN_FILE_2, 1, 2)
    write_dataset_to_csv_clinc(clinc_dataset["test"], TEST_FILE)


def setup_module():
    remove_files([TRAIN_FILE_1, TRAIN_FILE_2, TEST_FILE])
    download_clinc_dataset()


def test_distributed_udt_clinc(ray_two_node_cluster_config):
    # Import here so we don't get import errors collecting tests if ray isn't installed
    import thirdai.distributed_bolt as dist_bolt

    udt_model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        n_target_classes=150,
        integer_target=True,
    )

    udt_model.train_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[TRAIN_FILE_1, TRAIN_FILE_2],
    )

    assert (
        udt_model.evaluate(
            TEST_FILE, metrics=["categorical_accuracy"], return_metrics=True
        )["categorical_accuracy"]
        > 0.8
    )
