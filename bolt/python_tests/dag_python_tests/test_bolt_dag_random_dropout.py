from thirdai import bolt, dataset
import numpy as np
import pytest
from ..utils import gen_numpy_training_data

pytestmark = [pytest.mark.unit]


def test_random_dropout():
    num_classes = 10
    num_samples = 1000

    data, labels = gen_numpy_training_data(
        n_classes=num_classes, n_samples=num_samples, convert_to_bolt_dataset=True
    )
    input_layer = bolt.graph.Input(dim=num_classes)
    dense_hidden_layer = bolt.graph.FullyConnected(dim=100, activation="relu")(input_layer)
    sparse_hidden_layer = bolt.graph.FullyConnected(dim=256, activation="relu", sparsity=0.2, random_dropout=True)(dense_hidden_layer)
    output_layer = bolt.graph.FullyConnected(dim=num_classes, activation="softmax")(sparse_hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())

    train_config = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=3).silence()
    model.train(data, labels, train_config)


    predict_config = (
        bolt.graph.PredictConfig.make()
        .enable_sparse_inference()
        .with_metrics(["categorical_accuracy"])
        .return_activations()
        .silence()
    )
    metrics, activations =  model.predict(data, labels, predict_config)

    accuracy_returned = metrics["categorical_accuracy"]
    
    assert accuracy_returned > 0.8





