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
    ray_two_node_cluster_config,
    split_into_2,
    get_non_head_nodes,
)
from thirdai import bolt, dataset

pytestmark = [pytest.mark.distributed]

# TODO(Josh): This is quite a bit of duplicated code, but we can't easily share
# it until we change the structure of our python tests
def setup_module():
    import os

    path = "mnist_data"
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists("mnist_data/xaa") or not os.path.exists("mnist_data/xab"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")
        split_into_2(
            file_to_split="mnist",
            destination_file_1="mnist_data/part1",
            destination_file_2="mnist_data/part2",
        )

    if not os.path.exists("mnist_data/mnist.t"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")
        os.system("mv mnist.t mnist_data/")


def get_mnist_model():
    input_layer = bolt.nn.Input(dim=784)

    hidden_layer = bolt.nn.FullyConnected(dim=256, sparsity=0.5, activation="Relu")(
        input_layer
    )

    output_layer = bolt.nn.FullyConnected(dim=10, activation="softmax")(hidden_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

    return model


def get_distributed_mnist_model(request, ray_two_node_cluster_config, train_config):
    import thirdai.distributed_bolt as db

    model = get_mnist_model()

    # Because we explicitly specified the Ray working folder as this test
    # directory, but the current working directory where we downloaded mnist
    # may be anywhere, we give explicit paths for the mnist filenames
    train_sources = [
        db.DistributedSvmDatasetLoader(
            filename,
            batch_size=256,
        )
        for filename in [
            f"{os.getcwd()}/mnist_data/part1",
            f"{os.getcwd()}/mnist_data/part2",
        ]
    ]
    cluster_config, mini_cluster = ray_two_node_cluster_config(request.param)
    distributed_model = db.DistributedDataParallel(
        cluster_config=cluster_config,
        model=model,
        train_config=train_config,
        train_sources=train_sources,
    )
    return distributed_model, mini_cluster


def evaluated_distributed_mnist_model(distributed_model):
    check_models_are_same_on_first_two_nodes(distributed_model)

    eval_config = bolt.EvalConfig().with_metrics(["categorical_accuracy"]).silence()
    test_data, test_labels = dataset.load_bolt_svm_dataset(
        "mnist_data/mnist.t", batch_size=256
    )
    metrics = distributed_model.get_model().evaluate(
        test_data=test_data, test_labels=test_labels, eval_config=eval_config
    )
    return metrics


@pytest.fixture(scope="module")
def train_distributed_bolt_check(request, ray_two_node_cluster_config):

    train_config = bolt.TrainConfig(learning_rate=0.0001, epochs=3)
    distributed_model, _ = get_distributed_mnist_model(
        request, ray_two_node_cluster_config, train_config
    )
    distributed_model.train()

    metrics = evaluated_distributed_mnist_model(distributed_model)

    print(metrics)

    yield metrics


@pytest.fixture(scope="module")
def train_distributed_bolt_fault_tolerance(request, ray_two_node_cluster_config):

    train_config = bolt.TrainConfig(learning_rate=0.0001, epochs=1)
    distributed_model, mini_cluster = get_distributed_mnist_model(
        request, ray_two_node_cluster_config, train_config
    )
    distributed_model.train()
    node_to_kill = get_non_head_nodes(mini_cluster)[0]
    mini_cluster.remove_node(node_to_kill)
    # adding some waiting time
    import time

    time.sleep(2)
    mini_cluster.add_node(num_cpus=1)
    distributed_model.train()
    metrics = evaluated_distributed_mnist_model(distributed_model)

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


@pytest.mark.parametrize(
    "train_distributed_bolt_fault_tolerance", ["linear"], indirect=True
)
def test_distributed_fault_tolerance(train_distributed_bolt_fault_tolerance):
    import multiprocessing

    if multiprocessing.cpu_count() < 2:
        assert False, "not enough cpus for distributed training"

    assert train_distributed_bolt_fault_tolerance[0]["categorical_accuracy"] > 0.9
