import os

import pytest
from distributed_utils import remove_files
from thirdai import bolt
from thirdai.demos import download_clinc_dataset

pytestmark = [pytest.mark.distributed]

TRAIN_FILE_1 = "./clinc_train_0.csv"
TRAIN_FILE_2 = "./clinc_train_1.csv"
TEST_FILE = "./clinc_test.csv"


def setup_module():
    remove_files([TRAIN_FILE_1, TRAIN_FILE_2, TEST_FILE])
    download_clinc_dataset(num_training_files=2, clinc_small=True)


def get_clinc_udt_model(integer_target=False):
    udt_model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        n_target_classes=151,
        integer_target=integer_target,
    )
    return udt_model


# `ray_two_node_cluster_config` fixture added as parameter to start the mini_cluster
def test_distributed_udt_clinc(ray_two_node_cluster_config):

    udt_model = get_clinc_udt_model(integer_target=True)

    udt_model.train_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[f"{os.getcwd()}/{TRAIN_FILE_1}", f"{os.getcwd()}/{TRAIN_FILE_2}"],
        batch_size=256,
        epochs=6,
        learning_rate=0.01,
        metrics=["mean_squared_error"],
        verbose=True,
    )

    assert (
        udt_model.evaluate(
            f"{os.getcwd()}/{TEST_FILE}",
            metrics=["categorical_accuracy"],
            return_metrics=True,
        )["categorical_accuracy"]
        > 0.7
    )


# `ray_two_node_cluster_config` fixture added as parameter to start the mini_cluster
def test_non_integer_target_throws(ray_two_node_cluster_config):

    udt_model = get_clinc_udt_model(integer_target=False)

    with pytest.raises(
        ValueError,
        match="UDT with categorical target without integer_target=True cannot be "
        "trained in distributed "
        "setting. Please convert the categorical target column into "
        "integer target to train UDT in distributed setting.",
    ):
        # should fail, hence cluster_config and filenames are None
        udt_model.train_distributed(
            cluster_config=None,
            filenames=None,
        )


def test_temporal_relationships_throws(ray_two_node_cluster_config):
    udt_model = bolt.UniversalDeepTransformer(
        data_types={
            "userId": bolt.types.categorical(),
            "movieId": bolt.types.categorical(),
            "timestamp": bolt.types.date(),
        },
        temporal_tracking_relationships={"userId": ["movieId"]},
        target="movieId",
        n_target_classes=3,
        integer_target=True,
    )
    with pytest.raises(
        ValueError,
        match="UDT with temporal relationships cannot be trained in a distributed "
        "setting.",
    ):
        # should fail, hence cluster_config and filenames are None
        udt_model.train_distributed(
            cluster_config=None,
            filenames=None,
        )
