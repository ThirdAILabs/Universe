from thirdai import bolt, dataset
from utils import gen_training_data
import pytest
import numpy


def get_simple_concat_model(
    hidden_layer_top_dim,
    hidden_layer_bottom_dim,
    hidden_layer_top_sparsity,
    hidden_layer_bottom_sparsity,
    num_classes,
):

    input_layer = bolt.graph.Input(dim=num_classes)

    hidden_layer_top = bolt.graph.FullyConnected(
        dim=hidden_layer_top_dim,
        sparsity=hidden_layer_top_sparsity,
        activation="relu",
    )(input_layer)

    hidden_layer_bottom = bolt.graph.FullyConnected(
        dim=hidden_layer_bottom_dim,
        sparsity=hidden_layer_bottom_sparsity,
        activation="relu",
    )(input_layer)

    concate_layer = bolt.graph.Concatenate()([hidden_layer_top, hidden_layer_bottom])

    output_layer = bolt.graph.FullyConnected(dim=num_classes, activation="softmax")(
        concate_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


def run_simple_test(
    num_classes,
    hidden_layer_top_dim,
    hidden_layer_bottom_dim,
    hidden_layer_top_sparsity,
    hidden_layer_bottom_sparsity,
    num_training_samples,
    num_training_epochs=5,
    batch_size=64,
    learning_rate=0.001,
    accuracy_threshold=0.8,
):

    model = get_simple_concat_model(
        num_classes=num_classes,
        hidden_layer_top_dim=hidden_layer_top_dim,
        hidden_layer_bottom_dim=hidden_layer_bottom_dim,
        hidden_layer_top_sparsity=hidden_layer_top_sparsity,
        hidden_layer_bottom_sparsity=hidden_layer_bottom_sparsity,
    )

    train_data, train_labels = gen_training_data(
        n_classes=num_classes, n_samples=num_training_samples
    )

    train_config = bolt.graph.TrainConfig.makeConfig(
        learning_rate=learning_rate, epochs=num_training_epochs
    ).silence()

    metrics = model.train_np(
        train_data=train_data,
        batch_size=batch_size,
        train_labels=train_labels,
        train_config=train_config,
    )

    predict_config = (
        bolt.graph.PredictConfig.makeConfig()
        .withMetrics(["categorical_accuracy"])
        .silence()
    )

    metrics = model.predict_np(
        test_data=train_data,
        test_labels=train_labels,
        predict_config=predict_config,
    )

    assert metrics["categorical_accuracy"] >= accuracy_threshold


@pytest.mark.unit
def test_concat_dense_train():
    run_simple_test(
        num_classes=10,
        hidden_layer_top_dim=20,
        hidden_layer_bottom_dim=20,
        hidden_layer_top_sparsity=1,
        hidden_layer_bottom_sparsity=1,
        num_training_samples=10000,
    )


@pytest.mark.unit
def test_concat_sparse_train():
    run_simple_test(
        num_classes=100,
        hidden_layer_top_dim=100,
        hidden_layer_bottom_dim=100,
        hidden_layer_top_sparsity=0.1,
        hidden_layer_bottom_sparsity=0.1,
        num_training_samples=10000,
    )


@pytest.mark.unit
def test_concat_sparse_dense_train():
    run_simple_test(
        num_classes=100,
        hidden_layer_top_dim=100,
        hidden_layer_bottom_dim=100,
        hidden_layer_top_sparsity=0.1,
        hidden_layer_bottom_sparsity=1,
        num_training_samples=10000,
    )
