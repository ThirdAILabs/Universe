from thirdai import bolt, dataset
import pytest
import numpy as np

from ..utils import (
    gen_numpy_training_data,
    gen_random_weights_simple_network,
    gen_random_bias_simple_network,
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
    w1, w2 = gen_random_weights_simple_network(17, 5, 3)
    b1, b2 = gen_random_bias_simple_network(5, 3)
    hidden_layer = model.get_layer("fc_1")
    output_layer = model.get_layer("fc_2")
    hidden_layer.weights.set(w1)
    hidden_layer.biases.set(b1)
    output_layer.weights.set(w2)
    output_layer.biases.set(b2)


def test_bolt_dag_single_input_gradients():
    """
    For a given input and a fixed label on output, the INCREASE in activation of that label,
    when we add a small EPS to the input at each index seperately, should be in the
    same order as the input gradients. For example, let us have an input vector [1,0,0,0] and we choose output label as 1.
    If the input gradients are in the order <2,3,1,0> (the order is on the input indices), then the increase in activation for label 1
    should also be in same order, when we add small EPS at each position seperately.
    """
    model = build_dag_network()
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())
    set_network_weights_and_biases(model)
    numpy_inputs, numpy_labels = gen_numpy_training_data(
        n_classes=5, n_samples=100, convert_to_bolt_dataset=False
    )
    input_data = dataset.from_numpy(numpy_inputs, batch_size=10)
    labels = dataset.from_numpy(numpy_labels, batch_size=10)
    train_config = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=5).silence()
    model.train(input_data, labels, train_config)
    gradients = model.get_input_gradients(input_data, required_labels=numpy_labels)
    predict_config = (
        bolt.graph.PredictConfig.make()
        .enable_sparse_inference()
        .return_activations()
        .silence()
    )
    _, act = model.predict(input_data, None, predict_config=predict_config)
    """
    For every vector in input,we modify the vector at every position(by adding EPS), and we check the above assertion.
    """
    for input_num in range(len(numpy_inputs)):
        modified_numpy_vectors = []
        for i in range(len(numpy_inputs[input_num])):
            """
            We are making a copy because in python assign operation makes two variables to point
            to same address space, and we only want to modify one and keep the other same.
            """
            vec = np.array(numpy_inputs[input_num])
            vec[i] = vec[i] + 0.001
            modified_numpy_vectors.append(vec)
        modified_numpy_vectors = np.array(modified_numpy_vectors)
        modified_vectors = dataset.from_numpy(modified_numpy_vectors, batch_size=5)
        _, vecs_act = model.predict(
            modified_vectors, None, predict_config=predict_config
        )
        act_difference_at_required_label = [
            np.array(vec_act[numpy_labels[input_num]])
            - np.array(act[input_num][numpy_labels[input_num]])
            for vec_act in vecs_act
        ]
        assert np.array_equal(
            np.argsort(act_difference_at_required_label),
            np.argsort(gradients[input_num]),
        )
