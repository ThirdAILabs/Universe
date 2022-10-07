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
)
from thirdai import bolt, dataset

pytestmark = [pytest.mark.distributed]

try:
    import thirdai.distributed_bolt as db
except ImportError:
    pass


def get_simple_model(input_output_dim=10):
    input_layer = bolt.graph.Input(dim=input_output_dim)

    output_layer = bolt.graph.FullyConnected(dim=10, activation="softmax")(input_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


# Stream for learning f(x) = x
# 0 maps to a dataset where f(low) = low, 1 maps to a dataset where
# f(low + 1) = low + 1, ..., high - 1 maps to a dataset where
# f(high - 1) = high - 1. -> low. Everything else maps to none.
def get_dataset(dataset_id, low, high, dataset_size=1000, batch_size=256):
    if dataset_id + low >= high or dataset_id < 0:
        return None

    dataset_and_labels_np = (np.ones((dataset_size,)) * (low + dataset_id)).astype(
        "uint32"
    )
    dataset_and_labels = dataset.from_numpy(
        dataset_and_labels_np, batch_size=batch_size
    )
    return dataset_and_labels, dataset_and_labels


@pytest.fixture(scope="module")
def train_simple_streaming(ray_two_node_cluster_config):
    model = get_simple_model()

    train_sources = [
        db.GenericStreamingDataGenerator(lambda i: get_dataset(i, low=0, high=5)),
        db.GenericStreamingDataGenerator(lambda i: get_dataset(i, low=5, high=10)),
    ]
    train_config = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=10)
    distributed_model = db.DistributedDataParallel(
        cluster_config=ray_two_node_cluster_config,
        model=model,
        train_config=train_config,
        train_sources=train_sources,
    )
    distributed_model.train()

    check_models_are_same_on_first_two_nodes(distributed_model)

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )
    test_data_and_labels = dataset.from_numpy(
        np.random.randint(low=0, high=10, size=1000).astype("uint32"), batch_size=256
    )
    metrics = distributed_model.get_model().predict(
        test_data=test_data_and_labels,
        test_labels=test_data_and_labels,
        predict_config=predict_config,
    )

    print(metrics)

    yield metrics



@pytest.mark.skipif("ray" not in sys.modules, reason="requires the ray library")
def test_simple_streaming_with_distributed(train_simple_streaming):
    """
    Tests that a two worker cluster can learn f(x) = x, where the first worker
    gets data between 0 and 4 and the second worker gets data between 5 and 9.
    Uses a streaming data generator to pass in the training data. 
    """
    assert train_simple_streaming[0]["categorical_accuracy"] > 0.9
