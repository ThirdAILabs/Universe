from cgi import test
from thirdai import bolt
from ..utils import gen_numpy_training_data, get_simple_dag_model
import pytest

pytestmark = [pytest.mark.unit]

N_CLASSES = 10


def train_overfitted_model(train_data, train_labels):
    model = get_simple_dag_model(
        input_dim=N_CLASSES,
        hidden_layer_dim=1000,
        hidden_layer_sparsity=0.2,
        output_dim=N_CLASSES,
    )

    train_config = bolt.graph.TrainConfig.make(
        learning_rate=0.001, epochs=20
    ).with_metrics(["categorical_accuracy"])

    model.train(train_data, train_labels, train_config)

    return model


def train_early_stop_model(train_data, train_labels, valid_data, valid_labels):
    model = get_simple_dag_model(
        input_dim=N_CLASSES,
        hidden_layer_dim=1000,
        hidden_layer_sparsity=0.2,
        output_dim=N_CLASSES,
    )

    predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=20)
        .with_metrics(["categorical_accuracy"])
        .with_early_stop_validation(
            valid_data=valid_data,
            valid_labels=valid_labels,
            predict_config=predict_config,
            patience=3,
        )
    )

    model.train(train_data, train_labels, train_config)

    return model


def test_early_stop_validation():
    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=100, noise_std=0.3
    )
    valid_data, valid_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1000
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1000
    )

    overfitted_model = train_overfitted_model(train_data, train_labels)
    early_stop_model = train_early_stop_model(
        train_data, train_labels, valid_data, valid_labels
    )

    predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )

    overfitted_accuracy = overfitted_model.predict(
        test_data, test_labels, predict_config
    )[0]["categorical_accuracy"]

    early_stop_accuracy = early_stop_model.predict(
        test_data, test_labels, predict_config
    )[0]["categorical_accuracy"]

    print(early_stop_accuracy, overfitted_accuracy)
    assert early_stop_accuracy >= overfitted_accuracy
