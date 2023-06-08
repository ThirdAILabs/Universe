import math

import pytest
from thirdai import bolt

from utils import gen_numpy_training_data, get_simple_dag_model

pytestmark = [pytest.mark.unit]
N_CLASSES = 10


class GetEndingLearningRate(bolt.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.ending_lr = 0

    def on_train_end(self, model, train_state):
        self.ending_lr = train_state.learning_rate


def train_model_with_scheduler(
    epochs, base_learning_rate, schedule, batch_level_steps=False
):
    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=50, noise_std=0.3, batch_size_for_conversion=10
    )
    model = get_simple_dag_model(
        input_dim=N_CLASSES,
        hidden_layer_dim=2000,
        hidden_layer_sparsity=1.0,
        output_dim=N_CLASSES,
        output_activation="softmax",
        loss=bolt.nn.losses.CategoricalCrossEntropy(),
    )

    learning_rate_scheduler = bolt.callbacks.LearningRateScheduler(
        schedule=schedule, batch_level_steps=batch_level_steps
    )

    ending_lr_callback = GetEndingLearningRate()

    train_config = (
        bolt.TrainConfig(learning_rate=base_learning_rate, epochs=epochs)
        .with_metrics(["categorical_accuracy"])
        .with_callbacks([learning_rate_scheduler, ending_lr_callback])
    )

    model.train(train_data, train_labels, train_config)

    return ending_lr_callback.ending_lr


@pytest.mark.unit
def test_multiplicative_lr_scheduler():
    lr_schedule = bolt.callbacks.MultiplicativeLR(gamma=0.5)
    ending_lr = train_model_with_scheduler(
        base_learning_rate=0.01,
        epochs=2,
        schedule=lr_schedule,
    )

    assert math.isclose(ending_lr, 0.0025, rel_tol=1e-06)


@pytest.mark.unit
def test_exponential_lr_scheduler():
    lr_schedule = bolt.callbacks.ExponentialLR(gamma=0.5)
    ending_lr = train_model_with_scheduler(
        base_learning_rate=0.001,
        epochs=2,
        schedule=lr_schedule,
    )

    assert math.isclose(ending_lr, 0.00036787946, rel_tol=1e-06)


@pytest.mark.unit
def test_multistep_lr_scheduler():
    lr_schedule = bolt.callbacks.MultiStepLR(gamma=0.2, milestones=[1, 3])
    ending_lr = train_model_with_scheduler(
        base_learning_rate=0.001,
        epochs=4,
        schedule=lr_schedule,
    )
    assert math.isclose(ending_lr, 4e-05, rel_tol=1e-06)

##TODO: Gautam Sharma. Complete the test script for the LinearLR
@pytest.mark.unit
def test_linear_lr_scheduler():
    lr_schedule = bolt.callbacks.LinearLR()
    ending_lr = train_model_with_scheduler(
        base_learning_rate=0.001,
        epochs = 2,
        schedule=lr_schedule
    )
    assert math.isclose(ending_lr, 0.0004666666, rel_tol=1e-06)


@pytest.mark.unit
def test_custom_lr_scheduler():
    lr_schedule = bolt.callbacks.LambdaSchedule(
        lambda learning_rate, epoch: learning_rate * 0.1
    )

    ending_lr = train_model_with_scheduler(
        base_learning_rate=0.001,
        epochs=5,
        schedule=lr_schedule,
    )
    assert math.isclose(ending_lr, 1e-08, rel_tol=1e-06)


@pytest.mark.unit
def test_batch_level_steps():
    lr_schedule = bolt.callbacks.MultiStepLR(gamma=0.2, milestones=[1, 2])

    starting_lr = 0.001
    ending_lr = train_model_with_scheduler(
        base_learning_rate=starting_lr,
        epochs=1,
        schedule=lr_schedule,
        batch_level_steps=True,
    )

    assert math.isclose(ending_lr, starting_lr * 0.2 * 0.2, rel_tol=1e-06)
