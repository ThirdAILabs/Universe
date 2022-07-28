import math
import numpy as np
import pytest
from thirdai import bolt
from ..utils import gen_numpy_training_data


LEARNING_RATE = 0.001
BATCH_SIZE = 64

# Builds a DAG-based model to test setting and getting weights
def build_simple_model(num_classes, sparsity=0.5):
    input_layer = bolt.graph.Input(dim=num_classes)
    hidden_layer = bolt.graph.FullyConnected(dim=num_classes, activation="relu")(
        input_layer
    )
    output_layer = bolt.graph.FullyConnected(
        dim=100, sparsity=sparsity, activation="softmax"
    )(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


@pytest.mark.unit
def test_get_set_weights():
    """
    Tests that we can set and get weights for a specific node in the graph.
    This test ensures that substituting untrained weights with trained weights
    from the same model configuration results in comparable accuracy performances.
    """
    dataset_dim = 100
    train_data, train_labels = gen_numpy_training_data(
        n_classes=dataset_dim, n_samples=10000, batch_size_for_conversion=BATCH_SIZE
    )

    model = build_simple_model(num_classes=dataset_dim, sparsity=0.4)

    train_config = bolt.graph.TrainConfig.make(
        learning_rate=LEARNING_RATE, epochs=5
    ).silence()
    model.train(
        train_data=train_data, train_labels=train_labels, train_config=train_config
    )
    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )

    metrics = model.predict(
        test_data=train_data,
        test_labels=train_labels,
        predict_config=predict_config,
    )

    untrained_model = build_simple_model(num_classes=dataset_dim, sparsity=0.4)

    hidden_layer = model.get_layer("fc_1")
    output_layer = model.get_layer("fc_2")

    hidden_layer_weights = hidden_layer.get_weights()
    hidden_layer_biases = hidden_layer.get_biases()
    output_layer_weights = output_layer.get_weights()
    output_layer_biases = output_layer.get_biases()

    untrained_model.get_layer("fc_1").set_weights(
        new_weights=hidden_layer_weights
    ).set_biases(new_biases=hidden_layer_biases)

    untrained_model.get_layer("fc_2").set_weights(
        new_weights=output_layer_weights
    ).set_biases(new_biases=output_layer_biases)

    untrained_model_metrics = untrained_model.predict(
        test_data=train_data, test_labels=train_labels, predict_config=predict_config
    )

    assert math.isclose(
        untrained_model_metrics[0]["categorical_accuracy"],
        metrics[0]["categorical_accuracy"],
        rel_tol=0.001,
    )


@pytest.mark.unit
def test_bad_numpy_array_dim():
    dataset_dim = 100
    train_data, train_labels = gen_numpy_training_data(
        n_classes=dataset_dim, n_samples=10000, batch_size_for_conversion=BATCH_SIZE
    )

    model = build_simple_model(num_classes=dataset_dim, sparsity=0.4)

    train_config = bolt.graph.TrainConfig.make(
        learning_rate=LEARNING_RATE, epochs=5
    ).silence()
    model.train(
        train_data=train_data, train_labels=train_labels, train_config=train_config
    )

    untrained_model = build_simple_model(num_classes=dataset_dim, sparsity=0.4)

    hidden_layer = model.get_layer("fc_1")

    hidden_layer_weights = hidden_layer.get_weights()
    hidden_layer_biases = hidden_layer.get_biases()
    padded_weights = np.pad(hidden_layer_weights, pad_width=1)

    axis = 0
    bad_dim = dataset_dim + 2

    with pytest.raises(
        ValueError,
        match=f".*Expected dimension {axis} to be {dataset_dim} but received dimension {bad_dim}",
    ):
        untrained_model.get_layer("fc_1").set_weights(new_weights=padded_weights)

    with pytest.raises(
        ValueError,
        match=f".*Expected {hidden_layer_biases.ndim}D numpy array but received {padded_weights.ndim}D numpy array",
    ):
        untrained_model.get_layer("fc_1").set_biases(new_biases=padded_weights)
