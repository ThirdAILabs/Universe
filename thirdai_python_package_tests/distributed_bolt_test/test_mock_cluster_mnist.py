# Here we are mocking a cluster on a single machine
# without explicitly starting a ray cluster. We are
# testing both the communication circular and linear
# in the following tests.
# For reference: https://docs.ray.io/en/latest/ray-core/examples/testing-tips.html#tip-3-create-a-mini-cluster-with-ray-cluster-utils-cluster


import sys

import numpy as np
import pytest
from cluster_utils import (
    check_models_are_same_on_first_two_nodes,
    ray_two_node_cluster_config,
    split_into_2,
)
from thirdai import bolt, dataset

pytestmark = [pytest.mark.distributed]


try:
    import thirdai.distributed_bolt as db
except ImportError:
    pass


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
        split_into_2(file_to_split="mnist", destination_dir="mnist_data")

    if not os.path.exists("mnist_data/mnist.t"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")
        os.system("mv mnist.t mnist_data/")


def get_mnist_model():
    input_layer = bolt.graph.Input(dim=784)

    hidden_layer = bolt.graph.FullyConnected(dim=256, sparsity=0.5, activation="Relu")(
        input_layer
    )

    output_layer = bolt.graph.FullyConnected(dim=10, activation="softmax")(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


@pytest.fixture(scope="module")
def train_distributed_bolt_check(ray_two_node_cluster_config):
    model = get_mnist_model()
    data_sources = [
        {"train_file": "mnist_data/xaa", "batch_size": 256},
        {"train_file": "mnist_data/xab", "batch_size": 256},
    ]
    train_config = bolt.graph.TrainConfig.make(learning_rate=0.0001, epochs=1)

    distributed_model = db.DistributedDataParallel(
        cluster_config=ray_two_node_cluster_config,
        model=model,
        train_config=train_config,
        train_formats=["svm" for _ in range(len(data_sources))],
        train_data_sources=data_sources,
    )
    distributed_model.train()

    check_models_are_same_on_first_two_nodes(distributed_model)

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )
    test_data, test_labels = dataset.load_bolt_svm_dataset(
        "mnist_data/mnist.t", batch_size=256
    )
    metrics = distributed_model.get_model().predict(
        test_data=test_data, test_labels=test_labels, predict_config=predict_config
    )

    print(metrics)

    yield metrics


@pytest.mark.skipif("ray" not in sys.modules, reason="requires the ray library")
@pytest.mark.parametrize(
    "ray_two_node_cluster_config", ["linear", "circular"], indirect=True
)
def test_distributed_bolt_on_mock_cluster(train_distributed_bolt_check):
    import multiprocessing

    if multiprocessing.cpu_count() < 2:
        assert False, "not enough cpus for distributed training"

    assert train_distributed_bolt_check[0]["categorical_accuracy"] > 0.9
