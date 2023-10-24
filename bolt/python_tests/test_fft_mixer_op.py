import pytest
from thirdai import bolt

from utils import gen_numpy_training_data

N_CLASSES = 100


def build_model():
    input_layer = bolt.nn.Input(dim=N_CLASSES)

    # layer must not have sparsity for fft mixing
    hidden_layers = [
        bolt.nn.FullyConnected(
            dim=20,
            input_dim=input_layer.dim(),
        )
        for _ in range(10)
    ]

    concat = bolt.nn.Concatenate()(
        [hidden_layer(input_layer) for hidden_layer in hidden_layers]
    )
    mixing = bolt.nn.FFTMixer(10, 20)(concat)

    output_layer = bolt.nn.FullyConnected(
        dim=N_CLASSES,
        input_dim=mixing.dim(),
        activation="softmax",
    )(mixing)

    labels = bolt.nn.Input(dim=N_CLASSES)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    metric = bolt.train.metrics.CategoricalAccuracy(outputs=output_layer, labels=labels)

    return model, metric


@pytest.mark.unit
def test_fft_mixer_op():
    model, metric = build_model()

    train_data = gen_numpy_training_data(N_CLASSES)
    test_data = gen_numpy_training_data(N_CLASSES)

    trainer = bolt.train.Trainer(model)

    metrics = trainer.train(
        train_data=train_data,
        learning_rate=0.001,
        epochs=5,
        validation_data=test_data,
        validation_metrics={"acc": metric},
    )

    assert metrics["acc"][-1] >= 0.9  # Accuracy should be ~0.97-0.98.
