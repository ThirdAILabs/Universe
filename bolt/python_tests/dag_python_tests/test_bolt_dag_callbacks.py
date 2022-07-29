import pytest
from thirdai import bolt, dataset
from ..utils import gen_numpy_training_data
import math
import numpy as np


def get_simple_model(n_classes):
    input_layer = bolt.graph.Input(dim=n_classes)
    output_layer = bolt.graph.FullyConnected(dim=n_classes, activation="softmax")(
        input_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss(), print_when_done=False)
    return model


def train_simple_model(model, n_classes, n_samples, batch_size, epochs):
    data, labels = gen_numpy_training_data(
        n_classes=n_classes,
        n_samples=n_samples,
        convert_to_bolt_dataset=True,
        batch_size_for_conversion=batch_size,
    )
    train_cfg = bolt.graph.TrainConfig.make(
        learning_rate=0.001, epochs=epochs
    ).silence()
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
    # This test copies the weight and bias paramters from a single layer network. Then
    # it adds a callback which gets the gradients from the network and applies them to
    # the parameters. This is an approximation of the gradient descent on the parameters
    # (see comment below), and checks that getting and using gradients works correctly
    # and that we can use callbacks along with the state and parameters of the model.
    epochs = 5
    n_samples = 1000
    batch_size = 100
    n_classes = 10

    global model
    global layer_dims
    global weights
    global biases

    model = get_simple_model(n_classes)
    layer_dims = []
    weights = model.get_layer("fc_1").weights.copy()
    biases = model.get_layer("fc_1").biases.copy()

    def epoch_callback():
        global model
        global layer_dims

        dim = model.get_layer("fc_1").get_dim()
        layer_dims.append(dim)

    def batch_callback():
        global model
        global weights
        global biases

        w_grads = model.get_layer("fc_1").weight_gradients.get()
        b_grads = model.get_layer("fc_1").bias_gradients.get()
        # This is a different learning rate than used in train because we are using
        # simple gradient updates here instead of ADAM.
        #
        # Additionally this optimization is a bit hacky because the update strategy
        # is different meaning the parameters will diverge so the computed gradients
        # are not technically correct, but the dataset is simple enough that it still
        # works as a simple check.
        weights += 0.01 * w_grads
        biases += 0.01 * b_grads

    model.register_batch_callback(batch_callback)
    model.register_epoch_callback(epoch_callback)

    train_simple_model(
        model,
        n_classes=n_classes,
        n_samples=n_samples,
        batch_size=batch_size,
        epochs=epochs,
    )

    assert layer_dims == [n_classes for _ in range(epochs)]

    test_data, test_labels = gen_numpy_training_data(
        n_classes=n_classes, n_samples=n_samples, convert_to_bolt_dataset=False
    )

    outputs = np.matmul(test_data, weights.transpose())

    np_acc = np.mean(np.argmax(outputs, axis=1) == test_labels)

    assert np_acc > 0.95
