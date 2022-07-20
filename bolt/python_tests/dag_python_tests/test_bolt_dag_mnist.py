from ..test_mnist import ACCURACY_THRESHOLD, load_mnist
from ..utils import gen_training_data, get_simple_concat_model
from thirdai import bolt
import os
import math
import pytest

# Add an integration test marker for all tests in this file
pytestmark = [pytest.mark.integration]


LEARNING_RATE = 0.0001
BATCH_SIZE = 64


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

    hidden_layer = bolt.graph.FullyConnected(
        dim=20000,
        sparsity=0.01,
        activation="relu",
        sampling_config=bolt.SamplingConfig(
            num_tables=64, hashes_per_table=3, range_pow=9, reservoir_size=32
        ),
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(dim=10, activation="softmax")(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    train_data, train_labels, test_data, test_labels = load_mnist()

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.0001, epochs=3)
        .silence()
        .with_rebuild_hash_tables(3000)
        .with_reconstruct_hash_functions(10000)
    )

    metrics = model.train(
        train_data=train_data, train_labels=train_labels, train_config=train_config
    )

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )

    metrics = model.predict(
        test_data=test_data, test_labels=test_labels, predict_config=predict_config
    )

    assert metrics[0]["categorical_accuracy"] >= 0.9


# Builds a DAG-based model for MNIST with a sparse output layer
def build_sparse_output_layer_model(num_classes, sparsity=0.5):
    input_layer = bolt.graph.Input(dim=num_classes)
    hidden_layer = bolt.graph.FullyConnected(dim=num_classes, activation="relu")(
        input_layer
    )
    output_layer = bolt.graph.FullyConnected(
        dim=100, sparsity=sparsity, activation="softmax"
    )(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


def test_get_set_weights():
    """
    Tests that we can set and get weights for a specific node in the graph.
    This test ensures that substituting untrained weights with trained weights
    from the same model configuration results in comparable accuracy performances.
    """
    dataset_dim = 100
    train_data, train_labels = gen_training_data(n_classes=dataset_dim, n_samples=10000)

    model = build_sparse_output_layer_model(num_classes=dataset_dim, sparsity=0.4)

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=LEARNING_RATE, epochs=5)
        .with_batch_size(64)
        .silence()
    )
    model.train(
        train_data=train_data, train_labels=train_labels, train_config=train_config
    )
    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )

    metrics = model.predict(
        test_data=train_data,
        test_labels=train_labels,
        predict_config=predict_config,
    )

    untrained_model = build_sparse_output_layer_model(
        num_classes=dataset_dim, sparsity=0.4
    )

    hidden_layer = model.get_layer("fc_1")
    output_layer = model.get_layer("fc_2")

    hidden_layer_weights = hidden_layer.get_weights()
    hidden_layer_biases = hidden_layer.get_biases()
    output_layer_weights = output_layer.get_weights()
    output_layer_biases = output_layer.get_biases()

    untrained_model.get_layer("fc_1").set_weights(
        new_weights=hidden_layer_weights
    ).set_biases(new_biases=hidden_layer_biases)

    untrained_model.get_layer("fc_2").set_weights(
        new_weights=output_layer_weights
    ).set_biases(new_biases=output_layer_biases)

    untrained_model_metrics = untrained_model.predict(
        test_data=train_data, test_labels=train_labels, predict_config=predict_config
    )

    assert math.isclose(
        untrained_model_metrics[0]["categorical_accuracy"],
        metrics[0]["categorical_accuracy"],
        rel_tol=0.001,
    )
