import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]

import numpy as np
from thirdai import bolt, dataset

from utils import (
    gen_numpy_training_data,
    build_single_node_bolt_dag_model,
    build_simple_hidden_layer_model,
)

HIDDEN_DIM = 10
OUTPUT_DIM = 10
LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8

# A compressed dragon vector is exposed as a char array at this moment
# hence, it is not interpretable at Python end


def get_compressed_dragon_gradients(model, compression_density, seed_for_hashing):
    compressed_weight_grads = []
    layer1 = model.get_layer("fc_1")
    layer2 = model.get_layer("fc_2")

    compressed_weight_grads.append(
        layer1.weight_gradients.compress(
            compression_scheme="dragon",
            compression_density=compression_density,
            seed_for_hashing=seed_for_hashing,
            sample_population_size=50,
        )
    )

    compressed_weight_grads.append(
        layer2.weight_gradients.compress(
            compression_scheme="dragon",
            compression_density=compression_density,
            seed_for_hashing=seed_for_hashing,
            sample_population_size=50,
        )
    )

    return compressed_weight_grads


def set_compressed_dragon_gradients(model, compressed_weight_grads):
    layer1 = model.get_layer("fc_1")
    layer2 = model.get_layer("fc_2")
    layer1.weight_gradients.set(compressed_weight_grads[0])
    layer2.weight_gradients.set(compressed_weight_grads[1])
    return model


# We will get a compressed vector of gradients and then check whether the values are right
def test_get_set_values():
    model = build_simple_hidden_layer_model(input_dim=10, hidden_dim=10, output_dim=10)
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    first_layer = model.get_layer("fc_1")

    old_first_layer_biases = np.ravel(first_layer.biases.get())
    old_first_layer_weights = np.ravel(first_layer.weights.get())

    # getting the compressed gradients
    compressed_weights = first_layer.weights.compress(
        compression_scheme="dragon",
        compression_density=0.2,
        seed_for_hashing=1,
        sample_population_size=50,
    )

    compressed_biases = first_layer.biases.compress(
        compression_scheme="dragon",
        compression_density=0.2,
        seed_for_hashing=1,
        sample_population_size=10,
    )

    first_layer.weights.set(compressed_weights)
    first_layer.biases.set(compressed_biases)

    new_first_layer_biases = np.ravel(first_layer.biases.get())
    new_first_layer_weights = np.ravel(first_layer.weights.get())

    # checking whether the gradients are correct
    for i, values in enumerate(new_first_layer_weights):
        if values != 0:
            assert old_first_layer_weights[i] == new_first_layer_weights[i]

    for i, values in enumerate(new_first_layer_biases):
        if values != 0:
            assert old_first_layer_biases[i] == new_first_layer_biases[i]


# Instead of the earlier set function, set currently accepts a compressed vector
# if the compressed argument is True.
def test_concat_values():
    model = build_simple_hidden_layer_model(input_dim=10, hidden_dim=10, output_dim=10)
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    first_layer = model.get_layer("fc_1")
    old_first_layer_weights = np.ravel(first_layer.weights.get())

    # getting the compressed gradients
    compressed_weights = first_layer.weights.compress(
        compression_scheme="dragon",
        compression_density=0.2,
        seed_for_hashing=1,
        sample_population_size=50,
    )
    concatenated_weights = bolt.graph.ParameterReference.concat(
        [compressed_weights] * 2
    )
    first_layer.weights.set(concatenated_weights)
    new_first_layer_weights = np.ravel(first_layer.weights.get())

    for i, values in enumerate(new_first_layer_weights):
        if values != 0:
            assert 2 * old_first_layer_weights[i] == new_first_layer_weights[i]


# We compress the weight gradients of the model, and then reconstruct the weight
# gradients from the compressed dragon vector.
def test_compressed_dragon_vector_training():

    train_data, train_labels = gen_numpy_training_data(
        n_classes=10, n_samples=1000, convert_to_bolt_dataset=False
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=10, n_samples=100, convert_to_bolt_dataset=False
    )

    model = build_single_node_bolt_dag_model(
        train_data=train_data,
        train_labels=train_labels,
        sparsity=0.2,
        num_classes=10,
        learning_rate=LEARNING_RATE,
        hidden_layer_dim=30,
    )

    total_batches = model.numTrainingBatch()

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )

    for epochs in range(25):
        for batch_num in range(total_batches):
            model.calculateGradientSingleNode(batch_num)
            compressed_weight_grads = get_compressed_dragon_gradients(
                model,
                compression_density=0.25,
                seed_for_hashing=np.random.randint(100),
            )
            model = set_compressed_dragon_gradients(
                model, compressed_weight_grads=compressed_weight_grads
            )
            model.updateParametersSingleNode()

    model.finishTraining()
    acc = model.predict(
        test_data=dataset.from_numpy(test_data, batch_size=64),
        test_labels=dataset.from_numpy(test_labels, batch_size=64),
        predict_config=predict_config,
    )
    assert acc[0]["categorical_accuracy"] >= ACCURACY_THRESHOLD
