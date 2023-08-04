import pytest
from thirdai import bolt

from utils import gen_numpy_training_data

N_CLASSES = 100


def build_model():
    input_layer = bolt.nn.Input(dim=N_CLASSES)

    hidden_layer = bolt.nn.FullyConnected(
        dim=200,
        input_dim=input_layer.dim(),
        sparsity=0.4,
        activation="relu",
    )(input_layer)

    norm_layer = bolt.nn.LayerNorm()(hidden_layer)

    output_layer = bolt.nn.FullyConnected(
        dim=N_CLASSES,
        input_dim=norm_layer.dim(),
        activation="softmax",
    )(norm_layer)

    labels = bolt.nn.Input(dim=N_CLASSES)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    metric = bolt.train.metrics.CategoricalAccuracy(outputs=output_layer, labels=labels)

    return model, metric


@pytest.mark.unit
def test_layer_norm():
    model, metric = build_model()

    train_data = gen_numpy_training_data(N_CLASSES)
    test_data = gen_numpy_training_data(N_CLASSES)

    trainer = bolt.train.Trainer(model)

    metrics = trainer.train(
        train_data=train_data,
        learning_rate=0.05,
        epochs=5,
        validation_data=test_data,
        validation_metrics={"acc": metric},
    )

    assert metrics["acc"][-1] >= 0.9  # Accuracy should be ~0.97-0.98.
