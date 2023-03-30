# Here we are mocking a cluster on a single machine
# without explicitly starting a ray cluster. We are
# testing both the communication circular and linear
# in the following tests.
# For reference: https://docs.ray.io/en/latest/ray-core/examples/testing-tips.html#tip-3-create-a-mini-cluster-with-ray-cluster-utils-cluster


import os

import numpy as np
import pytest
from distributed_utils import mnist_distributed_split, ray_two_node_cluster_config
from download_dataset_fixtures import download_mnist_dataset
from thirdai import bolt as old_bolt
from thirdai import bolt_v2 as bolt

pytestmark = [pytest.mark.distributed]


def get_mnist_model():
    input_layer = bolt.nn.Input(dim=784)

    hidden_layer = bolt.nn.FullyConnected(
        dim=256, input_dim=input_layer.dim(), sparsity=0.5, activation="Relu"
    )(input_layer)

    output_layer = bolt.nn.FullyConnected(
        dim=10, input_dim=hidden_layer.dim(), activation="softmax"
    )(hidden_layer)

    labels = bolt.nn.Input(dim=output_layer.dim())

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    return model


def check_model_parameters_match(distributed_model):
    model_0 = distributed_model.get_model(0)
    model_1 = distributed_model.get_model(1)

    for op_0, op_1 in zip(model_0.ops(), model_1.ops()):
        assert np.allclose(op_0.weights, op_1.weights)
        assert np.allclose(op_0.biases, op_1.biases)


def train_distributed_bolt_v2(ray_cluster_config, train_files, test_file):
    import thirdai.distributed_bolt as db

    model = get_mnist_model()

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

    validation_context = db.ValidationContext(
        validation_source=db.DistributedSvmDatasetLoader(
            f"{os.getcwd()}/{test_file}", 256
        ),
        metrics=["categorical_accuracy"],
        sparse_inference=False,
        validation_frequency=118,
    )

    train_config = old_bolt.TrainConfig(learning_rate=0.0001, epochs=3)

    distributed_model = db.DistributedDataParallel(
        cluster_config=ray_cluster_config,
        model=model,
        train_config=train_config,
        train_sources=train_sources,
        validation_context=validation_context,
    )

    metrics = distributed_model.train()

    check_model_parameters_match(distributed_model)

    return metrics


# This test requires the Ray library, but we don't skip it if Ray isn't
# installed because if someone is running it part of the test may be if the
# Ray install is working at all. Marking it only with
# pytestmark.mark.distributed prevents it from running in our normal unit and
# integration test pipeline where ray isn't a dependency.
@pytest.mark.parametrize("comm_type", ["linear", "circular"])
def test_distributed_mnist_bolt_v2(
    comm_type, ray_two_node_cluster_config, mnist_distributed_split
):
    import multiprocessing

    if multiprocessing.cpu_count() < 2:
        assert False, "not enough cpus for distributed training"

    train_files, test_file = mnist_distributed_split

    metrics = train_distributed_bolt_v2(
        ray_cluster_config=ray_two_node_cluster_config(comm_type),
        train_files=train_files,
        test_file=test_file,
    )

    assert metrics["validation_metrics"][-1]["categorical_accuracy"] > 0.9
