import pytest
from thirdai import bolt

from utils import gen_numpy_training_data


def build_model(n_classes):
    input_layer = bolt.nn.Input(dim=n_classes)

    hidden_layer = bolt.nn.FullyConnected(
        dim=1000, input_dim=n_classes, sparsity=0.15, activation="relu"
    )(input_layer)
    output = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=1000, activation="softmax"
    )(hidden_layer)

    labels = bolt.nn.Input(dim=n_classes)
    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels)

    return bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])


@pytest.mark.unit
def test_freeze_hash_tables():
    n_classes = 100

    model = build_model(n_classes=n_classes)

    data = gen_numpy_training_data(n_classes=n_classes, n_samples=10000)

    trainer = bolt.train.Trainer(model)

    metrics1 = trainer.train(
        train_data=data,
        epochs=2,
        learning_rate=0.001,
        validation_data=data,
        validation_metrics=["categorical_accuracy"],
    )

    assert metrics1["val_categorical_accuracy"][-1] >= 0.8

    model.freeze_hash_tables()

    metrics2 = trainer.train(
        train_data=data,
        epochs=2,
        learning_rate=0.001,
        validation_data=data,
        validation_metrics=["categorical_accuracy"],
    )

    assert metrics2["val_categorical_accuracy"][-1] >= 0.9
    assert (
        metrics2["val_categorical_accuracy"][-1]
        >= metrics1["val_categorical_accuracy"][-1]
    )
