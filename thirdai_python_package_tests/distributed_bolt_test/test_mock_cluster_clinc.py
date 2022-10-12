import os

import numpy as np
import pytest
from cluster_utils import (
    check_models_are_same_on_first_two_nodes,
    ray_two_node_cluster_config,
    split_into_2,
)

# test_classifier_utils are not actually in this folder, but it works as long
# as you are running through pytest because pytest automatically adds all
# python folders not in a package to the path. Thus you can run this file like
# bin/python-test.sh -k "mock_cluster_clinc",
# but NOT like
# pytest thirdai_python_package_tests/distributed_bolt_test/test_mock_cluster_clinc.py
# See https://stackoverflow.com/questions/25827160/importing-correctly-with-pytest
# Relevant section:
# "The basepath may not match your intended basepath in which case the module
# will have a name that doesn't match what you would normally use. E.g., what
# you think of as geom.test.test_vector will actually be named just test_vector
# during the Pytest run because it found no __init__.py in src/geom/test/ and
# so added that directory to sys.path"
# This is definetely a hack, but we are already doing it in the compression
# module so it is fine to reduce duplication until we clean up our packaging
# structure.
from text_classifier_utils import *
from thirdai import bolt, dataset, deployment

pytestmark = [pytest.mark.integration, pytest.mark.release]


@pytest.fixture(scope="module")
def distributed_trained_text_classifier(
    saved_config, clinc_dataset, ray_two_node_cluster_config
):
    import thirdai.distributed_bolt as db

    num_classes, _ = clinc_dataset

    model_pipeline = deployment.ModelPipeline(
        config_path=saved_config,
        parameters={"size": "large", "output_dim": num_classes, "delimiter": ","},
    )

    path = "clinc_data"
    if not os.path.exists(path):
        os.makedirs(path)
    split_into_2(
        file_to_split=TRAIN_FILE,
        destination_file_1=f"clinc_data/part1",
        destination_file_2=f"clinc_data/part2",
    )

    # Because we explicitly specified the Ray working folder as this test
    # directory, but the current working directory where we downloaded clinc
    # may be anywhere, we give explicit paths for the mnist filenames
    train_config = bolt.graph.TrainConfig.make(epochs=5, learning_rate=0.01)
    wrapper = db.distribute_model_pipeline(
        cluster_config=ray_two_node_cluster_config,
        model_pipeline=model_pipeline,
        train_config=train_config,
        data_loaders=[
            (f"{os.getcwd()}/clinc_data/part1", 2560),
            (f"{os.getcwd()}/clinc_data/part2", 2560),
        ],
        max_in_memory_batches=10,
    )

    wrapper.train()

    model_pipeline.model = wrapper.get_model()

    return model_pipeline


@pytest.mark.parametrize("ray_two_node_cluster_config", ["linear"], indirect=True)
def test_distributed_classifer_accuracy(
    distributed_trained_text_classifier, clinc_dataset
):
    _, labels = clinc_dataset

    acc = np.mean(
        get_model_predictions(distributed_trained_text_classifier) == np.array(labels)
    )

    # Accuracy should be around 0.76 to 0.78.
    assert acc >= 0.7
