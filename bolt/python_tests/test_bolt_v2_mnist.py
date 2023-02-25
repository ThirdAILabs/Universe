import os

import pytest
from download_datasets import download_mnist
from thirdai import bolt as old_bolt
from thirdai import bolt_v2 as bolt
from thirdai import dataset

pytestmark = [pytest.mark.unit]


LEARNING_RATE = 0.0001


@pytest.fixture
def load_mnist_bolt_v2(download_mnist):
    train_file, test_file = download_mnist
    train_x, train_y = dataset.load_bolt_svm_dataset(train_file, 250)
    train_x = bolt.train.convert_dataset(train_x, dim=784)
    train_y = bolt.train.convert_dataset(train_y, dim=10)

    test_x, test_y = dataset.load_bolt_svm_dataset(test_file, 250)
    test_x = bolt.train.convert_dataset(test_x, dim=784)
    test_y = bolt.train.convert_dataset(test_y, dim=10)

    return (train_x, train_y), (test_x, test_y)


def test_bolt_on_mnist(load_mnist_bolt_v2):
    input_layer = bolt.nn.Input(dim=784)

    hidden_layer = bolt.nn.FullyConnected(
        dim=20000,
        input_dim=784,
        sparsity=0.01,
        activation="relu",
        sampling_config=old_bolt.nn.DWTASamplingConfig(
            num_tables=64, hashes_per_table=3, reservoir_size=32
        ),
        rebuild_hash_tables=12,
        reconstruct_hash_functions=40,
    )(input_layer)
    output = bolt.nn.FullyConnected(dim=10, input_dim=20000, activation="softmax")(
        hidden_layer
    )

    labels = bolt.nn.Input(dim=10)
    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    train_data, test_data = load_mnist_bolt_v2

    trainer = bolt.train.Trainer(model)

    history = trainer.train(
        train_data=train_data,
        epochs=3,
        learning_rate=0.0001,
        train_metrics={
            "loss": bolt.train.metrics.LossMetric(loss),
        },
        validation_data=test_data,
        validation_metrics={
            "loss": bolt.train.metrics.LossMetric(loss),
            "acc": bolt.train.metrics.CategoricalAccuracy(output, labels),
        },
        steps_per_validation=None,
        callbacks=[],
    )

    assert history["val_acc"][-1] >= 0.9
