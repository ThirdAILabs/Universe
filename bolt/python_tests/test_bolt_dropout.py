import pytest
from thirdai import bolt

from utils import gen_numpy_training_data


@pytest.mark.unit
def test_dropout_layer():
    n_classes = 20

    input_layer = bolt.nn.Input(dim=n_classes)
    hidden_layer = bolt.nn.FullyConnected(
        dim=100,
        sparsity=0.3,
        input_dim=n_classes,
        activation="relu",
        sampling_config=bolt.nn.RandomSamplingConfig(),
    )(input_layer)
    output_layer = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=hidden_layer.dim(), activation="softmax"
    )(hidden_layer)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        output_layer, labels=bolt.nn.Input(dim=n_classes)
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    train_data = gen_numpy_training_data(n_classes=n_classes, n_samples=2000)
    test_data = gen_numpy_training_data(n_classes=n_classes, n_samples=500)

    trainer = bolt.train.Trainer(model)

    metrics = trainer.train(
        train_data=train_data,
        learning_rate=0.001,
        epochs=3,
        validation_data=test_data,
        validation_metrics=["categorical_accuracy"],
    )

    assert metrics["val_categorical_accuracy"][-1] >= 0.8
