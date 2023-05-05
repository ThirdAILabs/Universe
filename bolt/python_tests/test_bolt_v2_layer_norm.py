import numpy as np
import pytest
from thirdai import bolt_v2 as bolt

from dataset import create_dataset

N_CLASSES = 100
INPUT_SHAPE = (20, 5, 2, N_CLASSES)


def build_model():
    input_layer = bolt.nn.Input(dims=INPUT_SHAPE[1:])

    hidden_layer = bolt.nn.FullyConnected(
        dim=200,
        input_dim=input_layer.dims()[-1],
        sparsity=0.4,
        activation="relu",
    )(input_layer)

    norm_layer = bolt.nn.LayerNorm()(hidden_layer)

    output_layer = bolt.nn.FullyConnected(
        dim=N_CLASSES,
        input_dim=norm_layer.dims()[-1],
        activation="softmax",
    )(norm_layer)

    labels = bolt.nn.Input(dims=INPUT_SHAPE[1:])

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    metric = bolt.train.metrics.CategoricalAccuracy(outputs=output_layer, labels=labels)

    return model, metric


@pytest.mark.unit
def test_op_sharing():
    model, metric = build_model()

    train_data = create_dataset(shape=INPUT_SHAPE, n_batches=5)
    test_data = create_dataset(shape=INPUT_SHAPE, n_batches=5)

    trainer = bolt.train.Trainer(model)

    metrics = trainer.train(
        train_data=train_data,
        learning_rate=0.05,
        epochs=5,
        validation_data=test_data,
        validation_metrics={"acc": metric},
    )

    assert metrics["acc"][-1] >= 0.9  # Accuracy should be ~0.97-0.98.

    for x, y in zip(test_data[0], test_data[1]):
        preds = model.forward(x, use_sparsity=False)[0]
        preds = np.argmax(preds.values, axis=-1)

        labels = y[0].indices
        labels = labels.reshape(labels.shape[:-1])

        acc = np.mean(preds == labels)

        assert acc >= 0.9
