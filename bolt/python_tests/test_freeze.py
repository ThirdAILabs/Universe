# %%
import numpy as np
import pytest
from thirdai import bolt

from utils import gen_numpy_training_data

# %%
pytestmark = [pytest.mark.unit]


def train_model(trainer, data):
    metrics1 = trainer.train(
        train_data=data,
        epochs=2,
        learning_rate=0.001,
        validation_data=data,
        validation_metrics=["categorical_accuracy"],
    )


@pytest.mark.unit
def test_freeze_layers():
    n_classes = 100

    input_layer = bolt.nn.Input(dim=n_classes)

    hidden_layer = bolt.nn.FullyConnected(
        dim=1000, input_dim=n_classes, sparsity=0.15, activation="relu"
    )(input_layer)
    output = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=1000, activation="softmax"
    )(hidden_layer)

    labels = bolt.nn.Input(dim=n_classes)

    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    data = gen_numpy_training_data(n_classes=n_classes, n_samples=10000)

    trainer = bolt.train.Trainer(model)

    train_model(trainer, data)

    before = {
        "weights": hidden_layer.weights.copy().flatten(),
        "biases": hidden_layer.biases.copy().flatten(),
    }

    # Freeze and train
    hidden_layer.trainable = False
    train_model(trainer, data)

    after = {
        "weights": hidden_layer.weights.copy().flatten(),
        "biases": hidden_layer.biases.copy().flatten(),
    }

    for key in ["weights", "biases"]:
        # The weights must remain the same
        assert np.all(np.equal(before[key], after[key]))

    # Undo freeze and train
    hidden_layer.trainable = True
    train_model(trainer, data)
    after = {
        "weights": hidden_layer.weights.copy().flatten(),
        "biases": hidden_layer.biases.copy().flatten(),
    }

    for key in ["weights", "biases"]:
        # The weights mustn't remain the same
        assert not np.all(np.equal(before[key], after[key]))


# %%
n_classes = 100
input_layer = bolt.nn.Input(dim=n_classes)

hidden_layer = bolt.nn.FullyConnected(
    dim=1000, input_dim=n_classes, sparsity=0.15, activation="relu"
)(input_layer)
# %%

# %%
