from thirdai import bolt, dataset
import os
import pytest
import numpy as np

from ..utils import gen_numpy_training_data

pytestmark = [pytest.mark.unit]


def build_dag_network():
    input_layer = bolt.graph.Input(dim=4)

    hidden_layer = bolt.graph.FullyConnected(
        dim=3,
        activation="relu",
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(dim=4, activation="softmax")(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    return model


def test_bolt_dag_input_gradients():
    model = build_dag_network()
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())
    numpy_inputs, numpy_labels = gen_numpy_training_data(
        n_classes=4, n_samples=4, convert_to_bolt_dataset=False
    )
    print(numpy_labels)
    input_data = dataset.from_numpy(numpy_inputs, batch_size=10)
    # labels = dataset.from_numpy(numpy_labels, batch_size=10)
    # train_config = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=5).silence()
    # model.train(input_data, labels, train_config)
    gradients = model.get_input_gradients(input_data, required_labels=numpy_labels)
    print(gradients)
    predict_config = (
        bolt.graph.PredictConfig.make()
        .enable_sparse_inference()
        .return_activations()
        .silence()
    )
    _, act = model.predict(input_data, None, predict_config=predict_config)
    for input_num in range(len(numpy_inputs)):
        modified_numpy_vectors = []
        for i in range(len(numpy_inputs[input_num])):
            """
            We are making a copy because in python assign operation makes two variables to point
            to same address space, and we only want to modify one and keep the other same.
            """
            vec = np.array(numpy_inputs[input_num])
            vec[i] = vec[i] + 0.01
            modified_numpy_vectors.append(vec)
        modified_numpy_vectors = np.array(modified_numpy_vectors)
        modified_vectors = dataset.from_numpy(modified_numpy_vectors, batch_size=10)
        _, vecs_act = model.predict(
            modified_vectors, None, predict_config=predict_config
        )
        act_difference_at_required_label = [
            np.array(vec_act[numpy_labels[input_num]])
            - np.array(act[input_num][numpy_labels[input_num]])
            for vec_act in vecs_act
        ]
        assert (
            np.argsort(act_difference_at_required_label).all()
            == np.argsort(gradients[input_num]).all()
        )
