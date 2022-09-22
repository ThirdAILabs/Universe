import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]

import numpy as np
from thirdai import bolt, dataset

from utils import (
    gen_numpy_training_data,
    build_single_node_bolt_dag_model,
    get_compressed_weight_gradients,
    set_compressed_weight_gradients,
)

LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8

# We compress the weight gradients of the model, and then reconstruct the weight
# gradients from the compressed dragon vector.
def test_compressed_count_sketch_training():
    num_sketches = 2
    compression_density = 0.2
    num_epochs = 30

    train_data, train_labels = gen_numpy_training_data(
        n_classes=10, n_samples=1000, convert_to_bolt_dataset=False
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=10, n_samples=100, convert_to_bolt_dataset=False
    )

    model = build_single_node_bolt_dag_model(
        train_data=train_data,
        train_labels=train_labels,
        sparsity=0.2,
        num_classes=10,
        learning_rate=LEARNING_RATE,
        hidden_layer_dim=50,
    )

    total_batches = model.numTrainingBatch()

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )
    for epochs in range(num_epochs):
        for batch_num in range(total_batches):
            model.calculateGradientSingleNode(batch_num)
            compressed_weight_grads = get_compressed_weight_gradients(
                model,
                compression_scheme="count_sketch",
                compression_density=compression_density,
                seed_for_hashing=np.random.randint(100),
                sample_population_size=num_sketches,
            )
            model = set_compressed_weight_gradients(
                model,
                compressed_weight_grads=compressed_weight_grads,
                compression_scheme="count_sketch",
            )
            model.updateParametersSingleNode()

    model.finishTraining()
    acc = model.predict(
        test_data=dataset.from_numpy(test_data, batch_size=64),
        test_labels=dataset.from_numpy(test_labels, batch_size=64),
        predict_config=predict_config,
    )
    assert acc[0]["categorical_accuracy"] >= ACCURACY_THRESHOLD
