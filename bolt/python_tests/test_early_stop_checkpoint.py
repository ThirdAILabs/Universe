from thirdai import bolt
from utils import gen_numpy_training_data, get_simple_dag_model
import pytest
import math
import os

pytestmark = [pytest.mark.unit]

N_CLASSES = 10
BEST_MODEL_SAVE_LOCATION = "./best.model"


def early_stop_train_config(metric_name, validation=False):
    metric_prefix = "val_" if validation else "train_"
    prefixed_metric = metric_prefix + metric_name

    return (
        bolt.graph.TrainConfig.make(learning_rate=0.01, epochs=20)
        .with_metrics([metric_name])
        .with_callbacks(
            [
                bolt.graph.callbacks.EarlyStopCheckpoint(
                    monitored_metric=prefixed_metric,
                    model_save_path=BEST_MODEL_SAVE_LOCATION,
                    patience=2,
                    min_delta=0,
                )
            ]
        )
    )


# this function trains a model with the early stop callback and evaluates the saved best model 
# and the last model on the given eval sets
def run_early_stop_test(
    train_data,
    train_labels,
    train_config,
    eval_data,
    eval_labels,
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

    model.train(train_data, train_labels, train_config)
    best_model = bolt.graph.Model.load(BEST_MODEL_SAVE_LOCATION)
    os.remove(BEST_MODEL_SAVE_LOCATION)

    predict_config = bolt.graph.PredictConfig.make().with_metrics([metric_name])

    last_model_score = model.predict(eval_data, eval_labels, predict_config)[0][
        metric_name
    ]

    early_stop_score = best_model.predict(eval_data, eval_labels, predict_config)[0][
        metric_name
    ]

    return last_model_score, early_stop_score


def test_early_stop_with_validation_accuracy():
    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=50, noise_std=0.3
    )
    valid_data, valid_labels = gen_numpy_training_data(  # CHANGED
        n_classes=N_CLASSES, n_samples=1000, noise_std=0.3
    )

    metric_name = "categorical_accuracy"

    train_config = early_stop_train_config(metric_name, validation=True)

    train_config = train_config.with_validation(
        validation_data=[valid_data],
        validation_labels=valid_labels,
        predict_config=bolt.graph.PredictConfig.make().with_metrics([metric_name]),
    )

    last_model_score, early_stop_score = run_early_stop_test(
        train_data=train_data,
        train_labels=train_labels,
        train_config=train_config,
        eval_data=valid_data,
        eval_labels=valid_labels,
        loss=bolt.CategoricalCrossEntropyLoss(),
        output_activation="softmax",
        metric_name=metric_name,
    )

    assert early_stop_score > last_model_score or math.isclose(
        early_stop_score, last_model_score, rel_tol=0.0001
    )


def test_early_stop_with_validation_loss():
    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=50, noise_std=0.3
    )
    valid_data, valid_labels = gen_numpy_training_data(  # CHANGED
        n_classes=N_CLASSES, n_samples=1000, noise_std=0.3
    )

    metric_name = "mean_squared_error"

    train_config = early_stop_train_config(metric_name, validation=True)

    train_config = train_config.with_validation(
        validation_data=[valid_data],
        validation_labels=valid_labels,
        predict_config=bolt.graph.PredictConfig.make().with_metrics([metric_name]),
    )

    last_model_score, early_stop_score = run_early_stop_test(
        train_data=train_data,
        train_labels=train_labels,
        train_config=train_config,
        eval_data=valid_data,
        eval_labels=valid_labels,
        loss=bolt.MeanSquaredError(),
        output_activation="linear",
        metric_name=metric_name,
    )
    assert early_stop_score < last_model_score or math.isclose(
        early_stop_score, last_model_score, rel_tol=0.0001
    )


def test_early_stop_with_train_accuracy():
    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=50, noise_std=0.3
    )

    metric_name = "categorical_accuracy"

    train_config = early_stop_train_config(metric_name, validation=False)

    last_model_score, early_stop_score = run_early_stop_test(
        train_data=train_data,
        train_labels=train_labels,
        train_config=train_config,
        eval_data=train_data,
        eval_labels=train_labels,
        loss=bolt.CategoricalCrossEntropyLoss(),
        output_activation="softmax",
        metric_name=metric_name,
    )

    assert early_stop_score > last_model_score or math.isclose(
        early_stop_score, last_model_score, rel_tol=0.0001
    )
