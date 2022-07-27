import pytest
from thirdai import bolt
from ..utils import gen_numpy_training_data
import numpy as np


def get_simple_model(n_classes):
    input_layer = bolt.graph.Input(dim=n_classes)
    output_layer = bolt.graph.FullyConnected(dim=n_classes, activation="softmax")(
        input_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())
    return model


def train_simple_model(model, n_classes, n_samples, batch_size, epochs):
    data, labels = gen_numpy_training_data(
        n_classes=n_classes,
        n_samples=n_samples,
        convert_to_bolt_dataset=True,
        batch_size_for_conversion=batch_size,
    )
    train_cfg = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=epochs)
    model.train(data, labels, train_cfg)


@pytest.mark.unit
def test_simple_dag_callbacks():

    epochs = 5
    n_samples = 1000
    batch_size = 100
    n_classes = 10

    global epoch_cnt
    epoch_cnt = 0
    def epoch_callback():
        global epoch_cnt
        epoch_cnt += 1

    global batch_cnt
    batch_cnt = 0
    def batch_callback():
        global batch_cnt
        batch_cnt += 1

    model = get_simple_model(n_classes)

    model.register_batch_callback(batch_callback)
    model.register_epoch_callback(epoch_callback)

    train_simple_model(
        model,
        n_classes=n_classes,
        n_samples=n_samples,
        batch_size=batch_size,
        epochs=epochs,
    )

    assert epoch_cnt == epochs
    assert batch_cnt == (epochs * n_samples / batch_size)


@pytest.mark.unit
def test_dag_callbacks_call_cpp_function():
    epochs = 5
    n_samples = 1000
    batch_size = 100
    n_classes = 10

    global global_model
    global layer_dims
    global layer_sparsities
    
    layer_dims = []
    layer_sparsities = []
    global_model = get_simple_model(n_classes)

    def epoch_callback():
        global global_model
        global layer_dims

        dim = global_model.get_layer("fc_1").get_dim()
        layer_dims.append(dim)

    def batch_callback():
        global global_model
        global layer_sparsities

        sparsity = global_model.get_layer("fc_1").get_sparsity()
        layer_sparsities.append(sparsity)

    global_model.register_batch_callback(batch_callback)
    global_model.register_epoch_callback(epoch_callback)

    train_simple_model(
        global_model,
        n_classes=n_classes,
        n_samples=n_samples,
        batch_size=batch_size,
        epochs=epochs,
    )

    assert len(layer_dims) == epochs
    assert len(layer_sparsities) == (epochs * n_samples / batch_size)
