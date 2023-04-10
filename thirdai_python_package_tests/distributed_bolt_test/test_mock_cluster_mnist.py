# Here we are mocking a cluster on a single machine
# without explicitly starting a ray cluster. We are
# testing both the communication circular and linear
# in the following tests.
# For reference: https://docs.ray.io/en/latest/ray-core/examples/testing-tips.html#tip-3-create-a-mini-cluster-with-ray-cluster-utils-cluster


import os
import sys

import numpy as np
import pytest
from distributed_utils import (
    check_models_are_same_on_first_two_nodes,
    mnist_distributed_split,
    ray_two_node_cluster_config,
)
from download_dataset_fixtures import download_mnist_dataset
from thirdai import bolt, dataset

pytestmark = [pytest.mark.distributed]


def get_mnist_model():
    input_layer = bolt.nn.Input(dim=784)

    hidden_layer = bolt.nn.FullyConnected(dim=256, sparsity=0.5, activation="Relu")(
        input_layer
    )

    output_layer = bolt.nn.FullyConnected(dim=10, activation="softmax")(hidden_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

    return model


@pytest.fixture(scope="module")
def train_distributed_bolt_check(
    request, ray_two_node_cluster_config, mnist_distributed_split
):
    import thirdai.distributed_bolt as db

    model = get_mnist_model()

    train_files, test_file = mnist_distributed_split

    # Because we explicitly specified the Ray working folder as this test
    # directory, but the current working directory where we downloaded mnist
    # may be anywhere, we give explicit paths for the mnist filenames
    train_sources = [
        db.DistributedSvmDatasetLoader(
            f"{os.getcwd()}/{filename}",
            batch_size=256,
        )
        for filename in train_files
    ]
    train_config = bolt.TrainConfig(learning_rate=0.0001, epochs=3)
    distributed_trainer = db.DistributedDataParallel(
        cluster_config=ray_two_node_cluster_config(request.param),
        model=model,
        train_config=train_config,
        train_sources=train_sources,
    )
    distributed_trainer.train(train_config.num_epochs)

    check_models_are_same_on_first_two_nodes(distributed_trainer)

    eval_config = bolt.EvalConfig().with_metrics(["categorical_accuracy"]).silence()
    test_data, test_labels = dataset.load_bolt_svm_dataset(test_file, batch_size=256)

    metrics = distributed_trainer.get_model().evaluate(
        test_data=test_data, test_labels=test_labels, eval_config=eval_config
    )

    print(metrics)

    yield metrics


# This test requires the Ray library, but we don't skip it if Ray isn't
# installed because if someone is running it part of the test may be if the
# Ray install is working at all. Marking it only with
# pytestmark.mark.distributed prevents it from running in our normal unit and
# integration test pipeline where ray isn't a dependency.
@pytest.mark.parametrize(
    "train_distributed_bolt_check", ["linear", "circular"], indirect=True
)
def test_distributed_mnist(train_distributed_bolt_check):
    import multiprocessing

    if multiprocessing.cpu_count() < 2:
        assert False, "not enough cpus for distributed training"

    assert train_distributed_bolt_check[0]["categorical_accuracy"] > 0.9
