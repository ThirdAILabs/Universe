from cgi import test
from thirdai import bolt
from utils import gen_numpy_training_data, get_simple_dag_model
import pytest
import os

pytestmark = [pytest.mark.unit]

N_CLASSES = 10


def train_models(train_data, train_labels, valid_data, valid_labels):
    model = get_simple_dag_model(
        input_dim=N_CLASSES,
        hidden_layer_dim=2000,
        hidden_layer_sparsity=1.0,
        output_dim=N_CLASSES,
    )

    predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )

    save_loc = "./best.model"

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=20)
        .with_metrics(["categorical_accuracy"])
        .with_callbacks(
            [
                bolt.graph.callbacks.EarlyStopCheckpoint(
                    validation_data=[valid_data],
                    validation_labels=valid_labels,
                    predict_config=predict_config,
                    best_model_save_location=save_loc,
                    patience=5,
                    min_delta=0,
                )
            ]
        )
    )

    model.train(train_data, train_labels, train_config)
    best_model = bolt.graph.Model.load(save_loc)
    os.remove(save_loc)

    return model, best_model


def test_early_stop_checkpoint():
    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=50, noise_std=0.3
    )
    valid_data, valid_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1000, noise_std=0.3
    )

    last_model, best_model = train_models(
        train_data, train_labels, valid_data, valid_labels
    )

    predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )

    last_model_accuracy = last_model.predict(valid_data, valid_labels, predict_config)[
        0
    ]["categorical_accuracy"]

    early_stop_accuracy = best_model.predict(valid_data, valid_labels, predict_config)[
        0
    ]["categorical_accuracy"]

    assert early_stop_accuracy >= last_model_accuracy
