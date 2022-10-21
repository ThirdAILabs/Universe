import os

import numpy as np
import pytest
from cluster_utils import (
    check_models_are_same_on_first_two_nodes,
    ray_two_node_cluster_config,
)

from thirdai import bolt, dataset, deployment
from thirdai.distributed_bolt import PandasLoader

pytestmark = [pytest.mark.distributed]

TRAIN_FILE = "clinc_data.csv"

@pytest.fixture(scope="module")
def distributed_trained_clinc(
    clinc_model, clinc_dataset, ray_two_node_cluster_config
):
    import thirdai.distributed_bolt as db

    num_classes, _ = clinc_dataset

    pandas_loaders = [
        PandasLoader(path=TRAIN_FILE, num_nodes=2, node_index=i, lines_per_load=500)
        for i in range(2)
    ]

    print(pandas_loaders[0].next().columns())
    exit(0)
    # DEFINE DATA PIPELINE

    wrapper.train()

    model_pipeline.model = wrapper.get_model()

    return model_pipeline


# def get_model_predictions(text_classifier):


@pytest.mark.parametrize("ray_two_node_cluster_config", ["linear"], indirect=True)
def test_distributed_classifer_accuracy(
    distributed_trained_clinc, clinc_labels
):
    _, labels = clinc_dataset

    acc = np.mean(
        get_model_predictions(distributed_trained_text_classifier) == np.array(labels)
    )

    # Accuracy should be around 0.76 to 0.78.
    assert acc >= 0.7
