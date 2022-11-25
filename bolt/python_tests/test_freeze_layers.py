import numpy as np
import pytest
from thirdai import bolt

from utils import gen_numpy_training_data

pytestmark = [pytest.mark.unit]


def test_freeze_layers():
    n_classes = 100

    input_dim = n_classes
    hidden_layer_dim = 1000
    hidden_layer_sparsity = 0.15
    output_dim = n_classes
    output_activation = "softmax"
    loss = bolt.nn.losses.CategoricalCrossEntropy()

    input_layer = bolt.nn.Input(dim=input_dim)

    hidden_layer = bolt.nn.FullyConnected(
        dim=hidden_layer_dim, sparsity=hidden_layer_sparsity, activation="relu"
    )(input_layer)

    output_layer = bolt.nn.FullyConnected(dim=output_dim, activation=output_activation)(
        hidden_layer
    )

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss)

    # Generate dataset.
    data, labels = gen_numpy_training_data(n_classes=n_classes, n_samples=10000)

    def train_evaluate(model, data, labels):
        # Train and predict before freezing hash tables.
        train_config = bolt.TrainConfig(learning_rate=0.001, epochs=2)
        model.train(data, labels, train_config)

        predict_config = (
            bolt.EvalConfig()
            .enable_sparse_inference()
            .with_metrics(["categorical_accuracy"])
        )

        test_metrics1 = model.evaluate(data, labels, predict_config)[0]
        assert test_metrics1["categorical_accuracy"] >= 0.8

    train_evaluate(model, data, labels)

    before = {
        "weights": hidden_layer.weights.copy().flatten(),
        "biases": hidden_layer.biases.copy().flatten(),
    }

    # Freeze and train
    hidden_layer.trainable(False)
    train_evaluate(model, data, labels)

    after = {
        "weights": hidden_layer.weights.copy().flatten(),
        "biases": hidden_layer.biases.copy().flatten(),
    }

    for key in ["weights", "biases"]:
        # The weights must remain the same
        assert np.all(np.equal(before[key], after[key]))

    # Undo freeze and train
    hidden_layer.trainable(True)
    train_evaluate(model, data, labels)
    after = {
        "weights": hidden_layer.weights.copy().flatten(),
        "biases": hidden_layer.biases.copy().flatten(),
    }

    for key in ["weights", "biases"]:
        # The weights mustn't remain the same
        assert not np.all(np.equal(before[key], after[key]))
