import os

import pytest
from distributed_utils import ray_two_node_cluster_config, remove_files
from thirdai import bolt
from thirdai.demos import download_clinc_dataset
import glob


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


# Tests that we can start a distributed job that trains for 0 epochs.
# This is currently necessary because running ray distributed jobs in the
# cibuildwheel docker container doesn't work: actors get killed randomly.
# We still want a  release test that tests distributed and makes sure licensing
# works, so this is the best we can do for now.
# TODO(Josh/Pratik): Look into getting ray working with cibuildwheel


def read_last_n_lines(file_path, n):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()[-n:]


@pytest.mark.release
def test_distributed_start(ray_two_node_cluster_config):
    udt_model = get_clinc_udt_model(integer_target=True)
    try:
        udt_model.train_distributed(
            cluster_config=ray_two_node_cluster_config("linear"),
            filenames=[
                f"{os.getcwd()}/{TRAIN_FILE_1}",
                f"{os.getcwd()}/{TRAIN_FILE_2}",
            ],
            epochs=0,
        )
    except Exception as err:
        print(f"{err}")
        folder_path = "/tmp/ray/session_latest/logs"
        n = 20
        file_types = ["*.log", "*.err", "*.out", "*.txt"]
        files = []

        for file_type in file_types:
            files.extend(glob.glob(os.path.join(folder_path, file_type)))

        for file_path in files:
            file_name = os.path.basename(file_path)
            print(f"File: {file_name}")
            print("Last 20 lines:")
            last_n_lines = read_last_n_lines(file_path, n)
            for line in last_n_lines:
                print(line.strip())
            print("\n" + "=" * 80 + "\n")


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
        epochs=1,
        learning_rate=0.02,
        metrics=["mean_squared_error"],
        verbose=True,
        max_in_memory_batches=10,
        validation=validation,
    )
    validation_metrics = training_and_validation_metrics["validation_metrics"]

    # check whether validation accuracy is increasing each time
    for metrics_next, metrics_prev in zip(validation_metrics[1:], validation_metrics):
        assert (
            metrics_next["categorical_accuracy"] > metrics_prev["categorical_accuracy"]
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
