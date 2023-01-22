import os

import pytest
from thirdai import bolt as old_bolt
from thirdai import bolt_v2 as bolt
from thirdai import dataset

# Add an integration test marker for all tests in this file
pytestmark = [pytest.mark.integration]


LEARNING_RATE = 0.0001


def load_mnist():
    train_x, train_y = dataset.load_bolt_svm_dataset("mnist", 250)
    train_x = bolt.train.convert_dataset(train_x, dim=784)
    train_y = bolt.train.convert_dataset(train_y, dim=10)

    test_x, test_y = dataset.load_bolt_svm_dataset("mnist.t", 250)
    test_x = bolt.train.convert_dataset(test_x, dim=784)
    test_y = bolt.train.convert_dataset(test_y, dim=10)

    return (train_x, train_y), (test_x, test_y)


def setup_module():
    if not os.path.exists("mnist"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")

    if not os.path.exists("mnist.t"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")


def test_bolt_on_mnist():
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

    loss = bolt.nn.losses.CategoricalCrossEntropy(output)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    train_data, test_data = load_mnist()

    trainer = bolt.train.Trainer(model)

    history = trainer.train(
        train_data=train_data,
        epochs=3,
        learning_rate=0.0001,
        train_metrics={
            "tensor_3": [bolt.train.metrics.LossMetric(loss)],
        },
        validation_data=test_data,
        validation_metrics={
            "tensor_3": [
                bolt.train.metrics.LossMetric(loss),
                bolt.train.metrics.CategoricalAccuracy(),
            ],
        },
        steps_per_validation=None,
        callbacks=[],
    )

    assert history["tensor_3"]["val_categorical_accuracy"][-1] >= 0.9
