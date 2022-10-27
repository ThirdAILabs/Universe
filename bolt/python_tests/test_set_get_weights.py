import math

import numpy as np
import pytest
from thirdai import bolt

from utils import gen_numpy_training_data, get_simple_dag_model

LEARNING_RATE = 0.001
BATCH_SIZE = 64


@pytest.mark.unit
def test_dag_get_set_weights():
    """
    Tests that we can set and get weights for a specific node in the graph.
    This test ensures that substituting untrained weights with trained weights
    from the same model configuration results in comparable accuracy performances.
    """
    dataset_dim = 100
    train_data, train_labels = gen_numpy_training_data(
        n_classes=dataset_dim, n_samples=10000, batch_size_for_conversion=BATCH_SIZE
    )

    model = get_simple_dag_model(
        input_dim=dataset_dim,
        hidden_layer_dim=dataset_dim,
        hidden_layer_sparsity=0.4,
        output_dim=dataset_dim,
    )

    train_config = bolt.TrainConfig(learning_rate=LEARNING_RATE, epochs=5).silence()
    model.train(
        train_data=train_data, train_labels=train_labels, train_config=train_config
    )
    predict_config = (
        bolt.PredictConfig().with_metrics(["categorical_accuracy"]).silence()
    )

    metrics = model.predict(
        test_data=train_data,
        test_labels=train_labels,
        predict_config=predict_config,
    )

    untrained_model = get_simple_dag_model(
        input_dim=dataset_dim,
        hidden_layer_dim=dataset_dim,
        hidden_layer_sparsity=0.4,
        output_dim=dataset_dim,
    )

    hidden_layer = model.get_layer("fc_1")
    output_layer = model.get_layer("fc_2")

    hidden_layer_weights = hidden_layer.weights.copy()
    hidden_layer_biases = hidden_layer.biases.copy()
    output_layer_weights = output_layer.weights.get()
    output_layer_biases = output_layer.biases.get()

    untrained_model.get_layer("fc_1").weights.set(hidden_layer_weights)
    untrained_model.get_layer("fc_1").biases.set(hidden_layer_biases)

    untrained_model.get_layer("fc_2").weights.set(output_layer_weights)
    untrained_model.get_layer("fc_2").biases.set(output_layer_biases)

    untrained_model_metrics = untrained_model.predict(
        test_data=train_data, test_labels=train_labels, predict_config=predict_config
    )

    assert math.isclose(
        untrained_model_metrics[0]["categorical_accuracy"],
        metrics[0]["categorical_accuracy"],
        rel_tol=0.00001,
    )


@pytest.mark.unit
def test_bad_numpy_array_dim():
    dataset_dim = 100

    model = get_simple_dag_model(
        input_dim=dataset_dim,
        hidden_layer_dim=dataset_dim,
        hidden_layer_sparsity=0.4,
        output_dim=dataset_dim,
    )

    hidden_layer = model.get_layer("fc_1")

    hidden_layer_weights = hidden_layer.weights.copy()
    hidden_layer_biases = hidden_layer.biases.copy()
    padded_weights = np.pad(hidden_layer_weights, pad_width=1)

    axis = 0
    bad_dim = dataset_dim + 2

    with pytest.raises(
        ValueError,
        match=f".*Expected dimension {axis} to be {dataset_dim} but received dimension {bad_dim}",
    ):
        model.get_layer("fc_1").weights.set(padded_weights)

    with pytest.raises(
        ValueError,
        match=f".*Expected {hidden_layer_biases.ndim}D numpy array but received {padded_weights.ndim}D numpy array",
    ):
        model.get_layer("fc_1").biases.set(padded_weights)
