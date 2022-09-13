from thirdai import bolt
from utils import gen_numpy_training_data, get_simple_dag_model
import pytest
import math

pytestmark = [pytest.mark.unit]
N_CLASSES = 10


def get_lr_scheduling_config(scheduling_primitive, parameters):
    return bolt.graph.callbacks.LRSchedulingConfig.make(
        scheduling_primitive
    ).with_parameters(parameters)


def get_model_with_scheduler(
    epochs, base_learning_rate, lr_scheduling_config, schedule, custom_scheduler=False
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
        loss=bolt.CategoricalCrossEntropyLoss(),
    )

    if custom_scheduler:
        learning_rate_scheduler = bolt.graph.callbacks.LearningRateScheduler(
            schedule=schedule
        )
    else:
        learning_rate_scheduler = bolt.graph.callbacks.LearningRateScheduler(
            config=lr_scheduling_config
        )
    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=base_learning_rate, epochs=epochs)
        .with_metrics(["categorical_accuracy"])
        .with_callbacks([learning_rate_scheduler])
    )

    model.train(train_data, train_labels, train_config)

    return learning_rate_scheduler, model


@pytest.mark.unit
def test_multiplicative_lr_scheduler():
    parameters = {"factor": [0.5]}
    lr_scheduling_config = get_lr_scheduling_config(
        scheduling_primitive="multiplicative-lr", parameters=parameters
    )
    learning_rate_scheduler, _ = get_model_with_scheduler(
        base_learning_rate=0.01,
        epochs=2,
        lr_scheduling_config=lr_scheduling_config,
        schedule=None,
    )

    assert math.isclose(learning_rate_scheduler.get_final_lr(), 0.005, rel_tol=1e-06)


@pytest.mark.unit
def test_exponential_lr_scheduler():
    parameters = {"gamma": [0.5]}
    lr_scheduling_config = get_lr_scheduling_config(
        scheduling_primitive="exponential-lr", parameters=parameters
    )
    learning_rate_scheduler, _ = get_model_with_scheduler(
        base_learning_rate=0.001,
        epochs=2,
        lr_scheduling_config=lr_scheduling_config,
        schedule=None,
    )

    assert math.isclose(
        learning_rate_scheduler.get_final_lr(), 0.00060653069, rel_tol=1e-06
    )


@pytest.mark.unit
def test_multistep_lr_scheduler():
    parameters = {"gamma": [0.2], "milestones": [2, 4]}
    lr_scheduling_config = get_lr_scheduling_config(
        scheduling_primitive="multistep-lr", parameters=parameters
    )
    learning_rate_scheduler, _ = get_model_with_scheduler(
        base_learning_rate=0.001,
        epochs=4,
        lr_scheduling_config=lr_scheduling_config,
        schedule=None,
    )

    assert math.isclose(learning_rate_scheduler.get_final_lr(), 0.00004, rel_tol=1e-06)


@pytest.mark.unit
def test_custom_lr_scheduler():

    lambda_scheduler = lambda learning_rate, epoch: learning_rate * 0.1

    learning_rate_scheduler, _ = get_model_with_scheduler(
        base_learning_rate=0.001,
        epochs=5,
        lr_scheduling_config=None,
        schedule=lambda_scheduler,
        custom_scheduler=True,
    )

    assert math.isclose(learning_rate_scheduler.get_final_lr(), 1e-07, rel_tol=1e-06)
