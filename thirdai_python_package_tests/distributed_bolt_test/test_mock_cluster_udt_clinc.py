import os

import pytest
from distributed_utils import ray_two_node_cluster_config, remove_files
from thirdai import bolt, data
from thirdai.demos import download_clinc_dataset

pytestmark = [pytest.mark.distributed]

TRAIN_FILE_1 = "./clinc_train_0.csv"
TRAIN_FILE_2 = "./clinc_train_1.csv"
TEST_FILE = "./clinc_test.csv"


def setup_module():
    remove_files([TRAIN_FILE_1, TRAIN_FILE_2, TEST_FILE])
    download_clinc_dataset(num_training_files=2, clinc_small=True)


def test_distributed_udt_clinc(ray_two_node_cluster_config):
    # Import here so we don't get import errors collecting tests if ray isn't installed
    import thirdai.distributed_bolt as dist_bolt

    udt_model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        n_target_classes=151,
        integer_target=True,
    )

    udt_model.train_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[f"{os.getcwd()}/{TRAIN_FILE_1}", f"{os.getcwd()}/{TRAIN_FILE_2}"],
        batch_size=256,
        epochs=5,
        learning_rate=0.01,
    )

    assert (
        udt_model.evaluate(
            f"{os.getcwd()}/{TEST_FILE}",
            metrics=["categorical_accuracy"],
            return_metrics=True,
        )["categorical_accuracy"]
        > 0.7
    )
