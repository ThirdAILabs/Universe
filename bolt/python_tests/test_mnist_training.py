import os

import pytest
from download_dataset_fixtures import download_mnist_dataset
from thirdai import bolt, dataset

pytestmark = [pytest.mark.unit]


LEARNING_RATE = 0.0001


@pytest.fixture
def load_mnist_bolt(download_mnist_dataset):
    train_file, test_file = download_mnist_dataset
    train_x, train_y = dataset.load_bolt_svm_dataset(train_file, 250)
    train_x = bolt.train.convert_dataset(train_x, dim=784)
    train_y = bolt.train.convert_dataset(train_y, dim=10)

    test_x, test_y = dataset.load_bolt_svm_dataset(test_file, 250)
    test_x = bolt.train.convert_dataset(test_x, dim=784)
    test_y = bolt.train.convert_dataset(test_y, dim=10)

    return (train_x, train_y), (test_x, test_y)


def test_bolt_on_mnist(load_mnist_bolt):
    input_layer = bolt.nn.Input(dim=784)

    hidden_layer = bolt.nn.FullyConnected(
        dim=20000,
        input_dim=784,
        sparsity=0.01,
        activation="relu",
        sampling_config=bolt.nn.DWTASamplingConfig(
            num_tables=64,
            hashes_per_table=3,
            range_pow=9,
            binsize=8,
            reservoir_size=32,
            permutations=8,
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

    train_data, test_data = load_mnist_bolt

    trainer = bolt.train.Trainer(model)

    history = trainer.train(
        train_data=train_data,
        learning_rate=0.0001,
        epochs=3,
        train_metrics=["loss"],
        validation_data=test_data,
        validation_metrics=["loss", "categorical_accuracy"],
    )

    # Accuracy should be ~0.93
    assert history["val_categorical_accuracy"][-1] >= 0.9

    history = trainer.validate(
        validation_data=test_data,
        validation_metrics={
            "sparse_acc": bolt.train.metrics.CategoricalAccuracy(output, labels)
        },
        use_sparsity=True,
    )

    # Accuracy should be ~0.82-0.83
    assert history["sparse_acc"][-1] >= 0.75
