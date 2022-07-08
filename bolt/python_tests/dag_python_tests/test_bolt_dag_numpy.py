from thirdai import bolt
import numpy as np
import pytest

from ..utils import gen_training_data

pytestmark = [pytest.mark.unit]


def build_train_and_predict(data, labels, num_classes, sparsity):

    input_layer = bolt.graph.Input(dim=num_classes)
    output_layer = bolt.graph.FullyConnected(
        dim=num_classes, activation="softmax", sparsity=sparsity
    )(input_layer)
    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=3)
        .with_batch_size(64)
        .silence()
    )
    model.train(data, labels, train_config)

    predict_config = (
        bolt.graph.PredictConfig.make()
        .enable_sparse_inference()
        .with_metrics(["categorical_accuracy"])
        .silence()
    )
    return model.predict(data, labels, predict_config)


def test_dense_numpy_output():
    num_classes = 100
    num_samples = 1000

    data, labels = gen_training_data(n_classes=num_classes, n_samples=num_samples)
    metrics, activations = build_train_and_predict(
        data, labels, num_classes, sparsity=1
    )

    assert activations.shape == (num_samples, num_classes)

    accuracy_returned = metrics["categorical_accuracy"]
    accuracy_computed = np.mean(np.argmax(activations, axis=1) == labels)

    assert accuracy_returned == accuracy_computed


def test_sparse_numpy_output():
    num_classes = 100
    sparsity = 0.1
    num_samples = 1000

    data, labels = gen_training_data(n_classes=num_classes, n_samples=num_samples)
    metrics, activations, active_neurons = build_train_and_predict(
        data, labels, num_classes, sparsity=sparsity
    )

    assert activations.shape == (num_samples, num_classes * sparsity)
    assert active_neurons.shape == (num_samples, num_classes * sparsity)
    assert np.all(active_neurons < num_classes)

    accuracy_returned = metrics["categorical_accuracy"]
    top_neurons_returned = active_neurons[
        np.arange(len(active_neurons)), np.argmax(activations, axis=1)
    ]
    accuracy_computed = np.mean(top_neurons_returned == labels)

    assert accuracy_returned == accuracy_computed
