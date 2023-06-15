import math

import pytest
from thirdai import bolt_v2 as bolt

from utils import gen_numpy_training_data

pytestmark = [pytest.mark.unit]
N_CLASSES = 10


def build_model():
    input_layer = bolt.nn.Input(dim=N_CLASSES)

    output_layer = bolt.nn.FullyConnected(
        dim=N_CLASSES,
        input_dim=input_layer.dim(),
        activation="softmax",
    )(input_layer)

    labels = bolt.nn.Input(dim=N_CLASSES)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    metric = bolt.train.metrics.CategoricalAccuracy(outputs=output_layer, labels=labels)

    return model, metric


def get_data(n_classes):
    x, y = gen_numpy_training_data(n_classes=n_classes)

    x = bolt.train.convert_dataset(x, dim=n_classes)
    y = bolt.train.convert_dataset(y, dim=n_classes)

    return x, y


class GetEndingLearningRate(bolt.train.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.ending_lr = 0

    def on_train_end(self):
        self.ending_lr = super().train_state.learning_rate


def train_model_with_scheduler(epochs, base_learning_rate, schedule):
    model, metric = build_model()

    train_data = get_data(N_CLASSES)
    test_data = get_data(N_CLASSES)

    ending_lr_callback = GetEndingLearningRate()

    trainer = bolt.train.Trainer(model)

    metrics = trainer.train(
        train_data=train_data,
        learning_rate=base_learning_rate,
        epochs=epochs,
        validation_data=test_data,
        validation_metrics={"acc": metric},
        callbacks=[schedule, ending_lr_callback],
    )

    return ending_lr_callback.ending_lr


@pytest.mark.unit
def test_linear_lr_scheduler():
    lr_schedule = bolt.train.callbacks.LinearLR()
    ending_lr = train_model_with_scheduler(
        epochs=2, base_learning_rate=0.001, schedule=lr_schedule
    )
    assert math.isclose(ending_lr, 0.00086666666, rel_tol=1e-06)


@pytest.mark.unit
def test_multi_step_lr():
    lr_schedule = bolt.train.callbacks.MultiStepLR(gamma=0.2, milestones=[1, 3])
    ending_lr = train_model_with_scheduler(
        epochs=4, base_learning_rate=0.001, schedule=lr_schedule
    )

    assert math.isclose(ending_lr, 4e-05, rel_tol=1e-06)
