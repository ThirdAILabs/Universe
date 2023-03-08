import os

import pytest
from thirdai import bolt

from utils import gen_numpy_training_data


def get_model_with_sparsification(n_classes):
    input_layer = bolt.nn.Input(n_classes)

    hidden = bolt.nn.FullyConnected(dim=400, activation="relu")(input_layer)

    hidden = bolt.nn.Sparsification(0.25)(hidden)

    output = bolt.nn.FullyConnected(dim=n_classes, activation="softmax")(hidden)

    model = bolt.nn.Model(inputs=[input_layer], output=output)
    model.compile(bolt.nn.losses.CategoricalCrossEntropy())

    return model


@pytest.mark.unit
def test_sparsification_layer():
    N_CLASSES = 100
    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=4_000
    )

    model = get_model_with_sparsification(N_CLASSES)

    model.train(
        train_data, train_labels, bolt.TrainConfig(epochs=2, learning_rate=0.001)
    )

    test_data, test_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1_000
    )

    eval_cfg = bolt.EvalConfig().with_metrics(["categorical_accuracy"])

    eval_metrics = model.evaluate(test_data, test_labels, eval_cfg)[0]
    # Accuracy is around 0.86 - 0.91
    assert eval_metrics["categorical_accuracy"] >= 0.8

    save_path = "saved_sparsification_model"
    model.save(save_path)
    model = bolt.nn.Model.load(save_path)
    os.remove(save_path)

    model.train(
        train_data, train_labels, bolt.TrainConfig(epochs=1, learning_rate=0.001)
    )

    eval_metrics = model.evaluate(test_data, test_labels, eval_cfg)[0]
    # Accuracy is around 0.99
    assert eval_metrics["categorical_accuracy"] >= 0.9
