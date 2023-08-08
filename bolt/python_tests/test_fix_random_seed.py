import numpy as np
import pytest
import thirdai
from thirdai import bolt, dataset


def build_model(fix_seed):
    if fix_seed:
        thirdai.set_seed(88)

    input_layer = bolt.nn.Input(dim=100)

    hidden_layer = bolt.nn.FullyConnected(
        dim=20,
        input_dim=100,
        activation="relu",
    )(input_layer)

    output = bolt.nn.FullyConnected(
        dim=50,
        input_dim=20,
        sparsity=0.5,
        activation="softmax",
    )(hidden_layer)

    labels = bolt.nn.Input(dim=50)
    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    return model


@pytest.mark.unit
def test_model_with_fixed_seed():
    model_1 = build_model(fix_seed=True)
    model_2 = build_model(fix_seed=True)

    for op_1, op_2 in zip(model_1.ops(), model_2.ops()):
        assert np.array_equal(op_1.weights, op_2.weights)
        assert np.array_equal(op_1.biases, op_2.biases)

    input_sample = [bolt.nn.Tensor(dataset.make_sparse_vector([0], [1.0]), 100)]

    output_1 = model_1.forward(input_sample, use_sparsity=True)[0]
    output_2 = model_2.forward(input_sample, use_sparsity=True)[0]

    assert set(output_1.active_neurons[0]) == set(output_2.active_neurons[0])


@pytest.mark.unit
def test_model_without_fixed_seed():
    model_1 = build_model(fix_seed=False)
    model_2 = build_model(fix_seed=False)

    for op_1, op_2 in zip(model_1.ops(), model_2.ops()):
        assert not np.array_equal(op_1.weights, op_2.weights)
        assert not np.array_equal(op_1.biases, op_2.biases)

    input_sample = [bolt.nn.Tensor(dataset.make_sparse_vector([0], [1.0]), 100)]

    output_1 = model_1.forward(input_sample, use_sparsity=True)[0]
    output_2 = model_2.forward(input_sample, use_sparsity=True)[0]

    assert not np.array_equal(output_1.active_neurons, output_2.active_neurons)
