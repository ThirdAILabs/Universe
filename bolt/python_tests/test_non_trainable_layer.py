import numpy as np
import pytest
from thirdai import bolt

from utils import build_simple_model, gen_numpy_training_data

pytestmark = [pytest.mark.unit]


def train_model(trainer, data):
    trainer.train(
        train_data=data,
        epochs=2,
        learning_rate=0.001,
        validation_data=data,
        validation_metrics=["categorical_accuracy"],
    )


@pytest.mark.unit
def test_non_trainable_layer():
    n_classes = 100

    model = build_simple_model(n_classes)

    data = gen_numpy_training_data(n_classes=n_classes, n_samples=10000)

    trainer = bolt.train.Trainer(model)

    train_model(trainer, data)

    ops = model.ops()

    before = {
        "weights": ops[0].weights.copy().flatten(),
        "biases": ops[0].biases.copy().flatten(),
    }

    # Freeze and train
    ops[0].trainable = False
    train_model(trainer, data)

    after = {
        "weights": ops[0].weights.copy().flatten(),
        "biases": ops[0].biases.copy().flatten(),
    }

    for key in ["weights", "biases"]:
        # The weights must remain the same
        assert np.allclose(before[key], after[key])

    # Undo freeze and train
    ops[0].trainable = True
    train_model(trainer, data)
    after = {
        "weights": ops[0].weights.copy().flatten(),
        "biases": ops[0].biases.copy().flatten(),
    }

    for key in ["weights", "biases"]:
        # The weights mustn't remain the same
        assert not np.allclose(before[key], after[key])
