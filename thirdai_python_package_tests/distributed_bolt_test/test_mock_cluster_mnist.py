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


def train_distributed_bolt_mnist(
    comm, epochs, ray_cluster_config, mnist_distributed_split
):
    import multiprocessing

    import thirdai.distributed_bolt as db

    if multiprocessing.cpu_count() < 2:
        assert False, "not enough cpus for distributed training"

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
    train_config = bolt.TrainConfig(learning_rate=0.0001, epochs=epochs)
    distributed_trainer = db.DistributedDataParallel(
        cluster_config=ray_cluster_config(comm),
        model=model,
        train_config=train_config,
        train_sources=train_sources,
    )
    for _ in range(train_config.num_epochs):
        while distributed_trainer.step():
            pass

        distributed_trainer.restart_data()

    check_models_are_same_on_first_two_nodes(distributed_trainer)

    eval_config = bolt.EvalConfig().with_metrics(["categorical_accuracy"]).silence()
    test_data, test_labels = dataset.load_bolt_svm_dataset(test_file, batch_size=256)

    metrics = distributed_trainer.get_model().evaluate(
        test_data=test_data, test_labels=test_labels, eval_config=eval_config
    )

    print(metrics)

    return metrics


def test_distributed_mnist_linear(ray_two_node_cluster_config, mnist_distributed_split):
    metrics = train_distributed_bolt_mnist(
        comm="linear",
        epochs=3,
        ray_cluster_config=ray_two_node_cluster_config,
        mnist_distributed_split=mnist_distributed_split,
    )
    assert metrics["categorical_accuracy"] >= 0.9


def test_distributed_mnist_circular(
    ray_two_node_cluster_config, mnist_distributed_split
):
    metrics = train_distributed_bolt_mnist(
        comm="circular",
        epochs=1,
        ray_cluster_config=ray_two_node_cluster_config,
        mnist_distributed_split=mnist_distributed_split,
    )
    assert metrics["categorical_accuracy"] >= 0.9
