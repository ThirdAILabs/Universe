import math

import pytest
from thirdai import bolt

from utils import gen_numpy_training_data, get_simple_dag_model

pytestmark = [pytest.mark.unit]
N_CLASSES = 10


def train_model_with_scheduler(
    epochs, base_learning_rate, lr_schedule, lambda_schedule, custom_scheduler=False
):

    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=50, noise_std=0.3
    )
    model = get_simple_dag_model(
        input_dim=N_CLASSES,
        hidden_layer_dim=2000,
        hidden_layer_sparsity=1.0,
        output_dim=N_CLASSES,
        output_activation="softmax",
        loss=bolt.nn.losses.CategoricalCrossEntropy(),
    )

    if custom_scheduler:
        learning_rate_scheduler = bolt.callbacks.LearningRateScheduler(
            schedule=lambda_schedule
        )
    else:
        learning_rate_scheduler = bolt.callbacks.LearningRateScheduler(
            schedule=lr_schedule
        )
    train_config = (
        bolt.TrainConfig(learning_rate=base_learning_rate, epochs=epochs)
        .with_metrics(["categorical_accuracy"])
        .with_callbacks([learning_rate_scheduler])
    )

    model.train(train_data, train_labels, train_config)

    return learning_rate_scheduler


@pytest.mark.unit
def test_multiplicative_lr_scheduler():

    lr_schedule = bolt.callbacks.MultiplicativeLR(gamma=0.5)
    learning_rate_scheduler = train_model_with_scheduler(
        base_learning_rate=0.01,
        epochs=2,
        lr_schedule=lr_schedule,
        lambda_schedule=None,
    )

    assert math.isclose(learning_rate_scheduler.get_final_lr(), 0.0025, rel_tol=1e-06)


@pytest.mark.unit
def test_exponential_lr_scheduler():
    lr_schedule = bolt.callbacks.ExponentialLR(gamma=0.5)
    learning_rate_scheduler = train_model_with_scheduler(
        base_learning_rate=0.001,
        epochs=2,
        lr_schedule=lr_schedule,
        lambda_schedule=None,
    )

    assert math.isclose(
        learning_rate_scheduler.get_final_lr(), 0.00036787946, rel_tol=1e-06
    )


@pytest.mark.unit
def test_multistep_lr_scheduler():
    lr_schedule = bolt.callbacks.MultiStepLR(gamma=0.2, milestones=[1, 3])
    learning_rate_scheduler = train_model_with_scheduler(
        base_learning_rate=0.001,
        epochs=4,
        lr_schedule=lr_schedule,
        lambda_schedule=None,
    )

    assert math.isclose(learning_rate_scheduler.get_final_lr(), 4e-05, rel_tol=1e-06)


@pytest.mark.unit
def test_custom_lr_scheduler():
    lr_schedule = bolt.callbacks.LambdaSchedule(
        lambda learning_rate, epoch: learning_rate * 0.1
    )

    learning_rate_scheduler = train_model_with_scheduler(
        base_learning_rate=0.001,
        epochs=5,
        lr_schedule=None,
        lambda_schedule=lr_schedule,
        custom_scheduler=True,
    )

    assert math.isclose(learning_rate_scheduler.get_final_lr(), 1e-08, rel_tol=1e-06)
