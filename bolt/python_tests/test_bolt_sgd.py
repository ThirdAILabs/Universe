import pytest
from thirdai import bolt

from utils import gen_numpy_training_data


# We paramaterize on the hidden sparsity so that the different types of sparse
# optimization methods are invoked.
@pytest.mark.unit
@pytest.mark.parametrize("hidden_sparsity", [1.0, 0.5])
def test_sgd(hidden_sparsity):
    n_classes = 20

    input_layer = bolt.nn.Input(dim=n_classes)
    hidden_layer = bolt.nn.FullyConnected(
        dim=100,
        sparsity=hidden_sparsity,
        input_dim=n_classes,
        activation="relu",
    )(input_layer)
    output_layer = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=hidden_layer.dim(), activation="softmax"
    )(hidden_layer)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        output_layer, labels=bolt.nn.Input(dim=n_classes)
    )

    model = bolt.nn.Model(
        inputs=[input_layer],
        outputs=[output_layer],
        losses=[loss],
        optimizer=bolt.nn.optimizers.SGD(),
    )

    train_data = gen_numpy_training_data(n_classes=n_classes, n_samples=2000)
    test_data = gen_numpy_training_data(n_classes=n_classes, n_samples=500)

    trainer = bolt.train.Trainer(model)

    metrics = trainer.train(
        train_data=train_data,
        learning_rate=0.5,
        epochs=5,
        validation_data=test_data,
        validation_metrics=["categorical_accuracy"],
        verbose=False,
    )

    assert metrics["val_categorical_accuracy"][-1] >= 0.9
