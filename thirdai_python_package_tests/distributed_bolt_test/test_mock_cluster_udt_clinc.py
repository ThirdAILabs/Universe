import os

import pytest
from distributed_utils import ray_two_node_cluster_config, remove_files
from thirdai import bolt, dataset
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
        options={"embedding_dimension": 128},
    )
    return udt_model


# Tests that we can start a distributed job that trains for 0 epochs.
@pytest.mark.release
def test_distributed_start(ray_two_node_cluster_config):
    udt_model = get_clinc_udt_model(integer_target=True)

    udt_model.train_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[f"{os.getcwd()}/{TRAIN_FILE_1}", f"{os.getcwd()}/{TRAIN_FILE_2}"],
        epochs=0,
    )


# `ray_two_node_cluster_config` fixture added as parameter to start the mini_cluster
def test_distributed_udt_clinc(ray_two_node_cluster_config):
    udt_model = get_clinc_udt_model(integer_target=True)

    validation = bolt.Validation(
        filename=f"{os.getcwd()}/{TEST_FILE}",
        metrics=["categorical_accuracy"],
        interval=10,
    )

    training_and_validation_metrics = udt_model.train_distributed(
        cluster_config=ray_two_node_cluster_config("linear"),
        filenames=[f"{os.getcwd()}/{TRAIN_FILE_1}", f"{os.getcwd()}/{TRAIN_FILE_2}"],
        batch_size=256,
        epochs=2,
        learning_rate=0.02,
        verbose=True,
        max_in_memory_batches=10,
        validation=validation,
        min_vecs_in_buffer=5000,
    )
    validation_metrics = training_and_validation_metrics["validation_metrics"]

    # check whether validation accuracy is increasing each time
    for metrics_next, metrics_prev in zip(validation_metrics[1:], validation_metrics):
        assert (
            metrics_next["val_categorical_accuracy"]
            > metrics_prev["val_categorical_accuracy"]
        )

    metrics = udt_model.evaluate(
        f"{os.getcwd()}/{TEST_FILE}",
        metrics=["categorical_accuracy"],
    )
    assert metrics["val_categorical_accuracy"][-1] > 0.7


# `ray_two_node_cluster_config` fixture added as parameter to start the mini_cluster
def test_non_integer_target_throws(ray_two_node_cluster_config):
    udt_model = get_clinc_udt_model(integer_target=False)

    with pytest.raises(
        ValueError,
        match="UDT with a categorical target cannot be trained in distributed "
        "setting without integer_target=True. Please convert the categorical "
        "target column into an integer target to train UDT in a distributed "
        "setting.",
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
