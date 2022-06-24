# Add an integration test marker for all tests in this file
from test_mnist import load_mnist, load_mnist_labels
from utils import train_network, build_sparse_hidden_layer_classifier
import numpy as np
from thirdai import bolt, dataset
import os
import pytest

pytestmark = [pytest.mark.integration]


LEARNING_RATE = 0.0001


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


def test_bolt_dag_on_mnist():
    input_layer = bolt.graph.Input(dim=784)

    hidden_layer = bolt.graph.FullyConnected(bolt.FullyConnected(
        dim=20000, sparsity=0.01, activation_function=bolt.ActivationFunctions.ReLU,
        sampling_config=bolt.SamplingConfig(num_tables=64, hashes_per_table=3, range_pow=9, reservoir_size=32)))
    hidden_layer(input_layer)

    output_layer = bolt.graph.FullyConnected(bolt.FullyConnected(
        dim=10, activation_function=bolt.ActivationFunctions.Softmax))
    output_layer(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    train_data, train_labels, test_data, test_labels = load_mnist()

    metrics = model.train(
        train_data=train_data,
        train_labels=train_labels,
        epochs=3,
        learning_rate=0.0001,
        rehash=3000,
        rebuild=10000
    )

    model.predict(
        test_data=test_data,
        test_labels=test_labels,
        metrics=["categorical_accuracy"]
    )

    metrics = model.predict(
        test_data, test_labels, metrics=["categorical_accuracy"], verbose=False
    )

    assert metrics["categorical_accuracy"] >= 0.9
