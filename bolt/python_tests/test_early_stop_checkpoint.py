import math
import os

import pytest
from thirdai import bolt

from utils import gen_numpy_training_data, get_simple_dag_model

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
    with_validation=True,
    with_validation_metrics=True,
    time_out=None,
):
    model = get_simple_dag_model(
        input_dim=N_CLASSES,
        hidden_layer_dim=2000,
        hidden_layer_sparsity=1.0,
        output_dim=N_CLASSES,
        output_activation=output_activation,
        loss=loss,
    )

    eval_config = bolt.EvalConfig()

    if with_validation_metrics:
        eval_config = eval_config.with_metrics([metric_name])

    save_loc = "./best.model"

    early_stop_callback = bolt.callbacks.EarlyStopCheckpoint(
        model_save_path=save_loc,
        monitored_metric=metric_name,
        patience=2,
        min_delta=0,
        time_out=time_out,
    )

    train_config = (
        bolt.TrainConfig(learning_rate=0.01, epochs=10)
        .with_metrics([metric_name])
        .with_callbacks([early_stop_callback])
    )

    if with_validation:
        train_config = train_config.with_validation(
            validation_data=[valid_data],
            validation_labels=valid_labels,
            eval_config=eval_config,
        )

    model.train(train_data, train_labels, train_config)
    best_model = bolt.nn.Model.load(save_loc)
    os.remove(save_loc)

    return model, best_model


# this method trains a model with the early stop callback and returns the
# validation scores for the best model saved and the last model
def run_early_stop_test(
    loss,
    output_activation,
    metric_name,
    with_validation=True,
    with_validation_metrics=True,
    time_out=None,
):
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
        with_validation,
        with_validation_metrics,
        time_out,
    )

    eval_config = bolt.EvalConfig().with_metrics([metric_name])

    last_model_score = last_model.evaluate(valid_data, valid_labels, eval_config)[0][
        metric_name
    ]

    early_stop_score = best_model.evaluate(valid_data, valid_labels, eval_config)[0][
        metric_name
    ]

    return last_model_score, early_stop_score


def test_early_stop_checkpoint_with_accuracy():
    last_model_score, early_stop_score = run_early_stop_test(
        loss=bolt.nn.losses.CategoricalCrossEntropy(),
        output_activation="softmax",
        metric_name="categorical_accuracy",
    )
    assert early_stop_score > last_model_score or math.isclose(
        early_stop_score, last_model_score, rel_tol=0.0001
    )


def test_early_stop_checkpoint_with_loss():
    last_model_score, early_stop_score = run_early_stop_test(
        loss=bolt.nn.losses.MeanSquaredError(),
        output_activation="linear",
        metric_name="mean_squared_error",
    )
    assert early_stop_score < last_model_score or math.isclose(
        early_stop_score, last_model_score, rel_tol=0.0001
    )


def test_throw_on_no_validation():
    with pytest.raises(
        ValueError,
        match=r"Could not find metric name 'mean_squared_error' in list of computed validation metrics.",
    ):
        run_early_stop_test(
            loss=bolt.nn.losses.MeanSquaredError(),
            output_activation="linear",
            metric_name="mean_squared_error",
            with_validation=False,
        )


def test_throw_on_no_validation_metrics():
    with pytest.raises(
        ValueError,
        match=r"Doing evaluation without metrics or activations is a no-op. Did you forget to specify this in the EvalConfig?",
    ):
        run_early_stop_test(
            loss=bolt.nn.losses.MeanSquaredError(),
            output_activation="linear",
            metric_name="mean_squared_error",
            with_validation_metrics=False,
        )


def test_throw_on_zero_patience():
    with pytest.raises(ValueError, match=r"Patience should be greater than 0."):
        bolt.callbacks.EarlyStopCheckpoint(
            model_save_path="dummy",
            patience=0,
        )


def test_throw_on_invalid_multiplier():
    with pytest.raises(ValueError, match=r"'lr_multiplier' should not be <= 0."):
        bolt.callbacks.EarlyStopCheckpoint(
            model_save_path="dummy",
            lr_multiplier=-1,
        )


def test_throw_on_invalid_time_out():
    with pytest.raises(ValueError, match=r"'time_out' cannot be negative."):
        bolt.callbacks.EarlyStopCheckpoint(
            model_save_path="dummy",
            time_out=-1,
        )


def test_invalid_compare_against_string():
    with pytest.raises(
        ValueError, match=r"'compare_against' should be one of 'best' or 'prev'."
    ):
        bolt.callbacks.EarlyStopCheckpoint(
            model_save_path="dummy", compare_against="dummy"
        )


def test_time_out():
    run_early_stop_test(
        loss=bolt.nn.losses.MeanSquaredError(),
        output_activation="linear",
        metric_name="mean_squared_error",
        time_out=0.00001,
    )
