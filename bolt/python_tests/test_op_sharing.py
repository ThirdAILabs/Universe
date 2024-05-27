import pytest
from thirdai import bolt

from utils import gen_numpy_training_data

N_CLASSES = 100


def build_model(hidden_op, output_op):
    input_layer = bolt.nn.Input(dim=N_CLASSES)

    hidden_layer = hidden_op(input_layer)

    output_layer = output_op(hidden_layer)

    labels = bolt.nn.Input(dim=N_CLASSES)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    metric = bolt.train.metrics.CategoricalAccuracy(outputs=output_layer, labels=labels)

    return model, metric


@pytest.mark.unit
def test_op_sharing():
    hidden_op = bolt.nn.FullyConnected(
        dim=200,
        input_dim=N_CLASSES,
        sparsity=0.4,
        activation="relu",
    )

    output_op = bolt.nn.FullyConnected(
        dim=N_CLASSES,
        input_dim=200,
        activation="softmax",
    )

    original_model, original_metric = build_model(hidden_op, output_op)

    train_data = gen_numpy_training_data(N_CLASSES)
    test_data = gen_numpy_training_data(N_CLASSES)

    original_trainer = bolt.train.Trainer(original_model)

    original_metrics = original_trainer.train(
        train_data=train_data,
        learning_rate=0.05,
        epochs=5,
        validation_data=test_data,
        validation_metrics={"acc": original_metric},
    )

    new_model, new_metric = build_model(hidden_op, output_op)

    new_trainer = bolt.train.Trainer(new_model)

    new_metrics = new_trainer.validate(
        validation_data=test_data, validation_metrics={"acc": new_metric}
    )

    assert original_metrics["acc"][-1] == new_metrics["acc"][-1]
    assert new_metrics["acc"][-1] >= 0.9  # Accuracy should be ~0.97-0.98.
