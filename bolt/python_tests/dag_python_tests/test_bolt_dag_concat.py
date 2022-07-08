from thirdai import bolt, dataset
from ..utils import gen_training_data, get_simple_concat_model
import pytest
import numpy


# TODO(josh): Refactor this test once we have exposed support for multiple
# inputs to split up the input so that there isn't a complete set of information
# that's being provided to both layers in the concatenation
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

    train_config = bolt.graph.TrainConfig.make(
        learning_rate=learning_rate, epochs=num_training_epochs
    ).silence()

    metrics = model.train_np(
        train_data=train_data,
        batch_size=batch_size,
        train_labels=train_labels,
        train_config=train_config,
    )

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )

    metrics = model.predict(
        test_data=train_data,
        test_labels=train_labels,
        predict_config=predict_config,
    )

    assert metrics[0]["categorical_accuracy"] >= accuracy_threshold


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


test_concat_dense_train()
test_concat_sparse_train()
test_concat_sparse_dense_train()
