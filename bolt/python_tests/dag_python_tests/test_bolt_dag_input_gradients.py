from thirdai import bolt, dataset
import pytest
import numpy as np

from ..utils import (
    assert_activation_difference_and_gradients_in_same_order,
    gen_numpy_training_data,
    gen_random_weights_simple_network,
    gen_random_bias_simple_network,
    get_perturbed_dataset,
)

pytestmark = [pytest.mark.unit]


def build_dag_network():
    input_layer = bolt.graph.Input(dim=5)

    hidden_layer = bolt.graph.FullyConnected(
        dim=3,
        activation="relu",
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(dim=5, activation="softmax")(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    return model


def set_network_weights_and_biases(model):
    w1, w2 = gen_random_weights_simple_network(
        input_output_layer_dim=5, hidden_layer_dim=3
    )
    b1, b2 = gen_random_bias_simple_network(output_layer_dim=5, hidden_layer_dim=3)
    hidden_layer = model.get_layer("fc_1")
    output_layer = model.get_layer("fc_2")
    hidden_layer.weights.set(w1)
    hidden_layer.biases.set(b1)
    output_layer.weights.set(w2)
    output_layer.biases.set(b2)


def build_compile_and_set_weights_and_biases():
    model = build_dag_network()
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())
    set_network_weights_and_biases(model)
    return model


def model_train_and_predict(model, numpy_inputs, numpy_labels, batch_size):
    input_data = dataset.from_numpy(numpy_inputs, batch_size=batch_size)
    labels = dataset.from_numpy(numpy_labels, batch_size=batch_size)
    train_config = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=5).silence()
    model.train(input_data, labels, train_config)
    predict_config = (
        bolt.graph.PredictConfig.make()
        .enable_sparse_inference()
        .return_activations()
        .silence()
    )
    _, act = model.predict(input_data, None, predict_config=predict_config)
    return input_data, act, predict_config


def get_perturbed_dataset_and_assert(
    model, numpy_input, predict_config, numpy_label, gradients, act
):
    perturbed_dataset = get_perturbed_dataset(numpy_input)
    _, perturbed_activations = model.predict(
        perturbed_dataset, None, predict_config=predict_config
    )
    assert_activation_difference_and_gradients_in_same_order(
        perturbed_activations,
        numpy_label,
        gradients,
        act,
    )


def test_bolt_dag_single_input_node_gradients_single():
    """
    For a given input and a fixed label on output, the INCREASE in activation of that label,
    when we add a small EPS to the input at each index seperately, should be in the
    same order as the input gradients. For example, let us have an input vector [1,0,0,0] and we choose output label as 1.
    If the input gradients are in the order <2,3,1,0> (the order is on the input indices), then the increase in activation for label 1
    should also be in same order, when we add small EPS at each position seperately.
    """
    model = build_compile_and_set_weights_and_biases()
    numpy_inputs, numpy_labels = gen_numpy_training_data(
        n_classes=5, n_samples=100, convert_to_bolt_dataset=False
    )
    """
    Here we are passing batch size as 1, because that way we can easily acesss single input and pass it to the get_input_gradients_single method.
    """
    input_data, act, predict_config = model_train_and_predict(
        model, numpy_inputs, numpy_labels, batch_size=1
    )
    for i in range(len(numpy_inputs)):
        gradients = model.get_input_gradients_single(
            input_data[i], neuron_to_explain=numpy_labels[i]
        )
        """
        For every vector in input,we modify the vector at every position(by adding EPS), and we check assertion discussed at start of the function.
        """
        get_perturbed_dataset_and_assert(
            model,
            numpy_inputs[i],
            predict_config,
            numpy_labels[i],
            gradients,
            act[i],
        )
