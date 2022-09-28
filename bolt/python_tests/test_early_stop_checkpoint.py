from thirdai import bolt
from utils import gen_numpy_training_data, get_simple_dag_model
import pytest
import math
import os

pytestmark = [pytest.mark.unit]

N_CLASSES = 10


def train_models(
    train_data,
    train_labels,
    valid_data,
    valid_labels,
    loss,
    output_activation,
    metric_name,
):
    model = get_simple_dag_model(
        input_dim=N_CLASSES,
        hidden_layer_dim=2000,
        hidden_layer_sparsity=1.0,
        output_dim=N_CLASSES,
        output_activation=output_activation,
        loss=loss,
    )

    predict_config = bolt.graph.PredictConfig.make().with_metrics([metric_name])

    save_loc = "./best.model"

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.01, epochs=20)
        .with_metrics([metric_name])
        .with_validation(
            validation_data=[valid_data],
            validation_labels=valid_labels,
            predict_config=predict_config,
        )
        .with_callbacks(
            [
                bolt.graph.callbacks.EarlyStopCheckpoint(
                    monitored_metric=metric_name,
                    model_save_path=save_loc,
                    patience=2,
                    min_delta=0,
                )
            ]
        )
    )

    model.train(train_data, train_labels, train_config)
    best_model = bolt.graph.Model.load(save_loc)
    os.remove(save_loc)

    return model, best_model


# this method trains a model with the early stop callback and returns the
# validation scores for the best model saved and the last model
def run_early_stop_test(loss, output_activation, metric_name):
    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1000, noise_std=0.3
    )
    valid_data, valid_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1000, noise_std=0.3
    )

    last_model, best_model = train_models(
        train_data,
        train_labels,
        valid_data,
        valid_labels,
        loss,
        output_activation,
        metric_name,
    )

    predict_config = bolt.graph.PredictConfig.make().with_metrics([metric_name])

    last_model_score = last_model.predict(valid_data, valid_labels, predict_config)[0][
        metric_name
    ]

    early_stop_score = best_model.predict(valid_data, valid_labels, predict_config)[0][
        metric_name
    ]

    return last_model_score, early_stop_score


def test_early_stop_checkpoint_with_accuracy():
    last_model_score, early_stop_score = run_early_stop_test(
        loss=bolt.CategoricalCrossEntropyLoss(),
        output_activation="softmax",
        metric_name="categorical_accuracy",
    )
    assert early_stop_score > last_model_score or math.isclose(
        early_stop_score, last_model_score, rel_tol=0.0001
    )


def test_early_stop_checkpoint_with_loss():
    last_model_score, early_stop_score = run_early_stop_test(
        loss=bolt.MeanSquaredError(),
        output_activation="linear",
        metric_name="mean_squared_error",
    )
    assert early_stop_score < last_model_score or math.isclose(
        early_stop_score, last_model_score, rel_tol=0.0001
    )