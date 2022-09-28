import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]

import numpy as np
from thirdai import bolt, dataset

from utils import (
    build_simple_hidden_layer_model,
    compressed_training,
)

INPUT_DIM = 10
HIDDEN_DIM = 10
OUTPUT_DIM = 10
LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8

# A compressed vector is exposed as a char array
# and hence, it is not interpretable at Python end


def test_get_set_values_dragon_vector():
    model = build_simple_hidden_layer_model(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM
    )
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    first_layer = model.get_layer("fc_1")

    old_first_layer_biases = first_layer.biases.copy().flatten()
    old_first_layer_weights = first_layer.weights.copy().flatten()

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

    new_first_layer_biases = first_layer.biases.copy().flatten()
    new_first_layer_weights = first_layer.weights.copy().flatten()

    # checking whether the gradients are correct
    for i, values in enumerate(new_first_layer_weights):
        if values != 0:
            assert old_first_layer_weights[i] == new_first_layer_weights[i]

    for i, values in enumerate(new_first_layer_biases):
        if values != 0:
            assert old_first_layer_biases[i] == new_first_layer_biases[i]


def test_concat_values_dragon_vector():
    model = build_simple_hidden_layer_model(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM
    )
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    first_layer = model.get_layer("fc_1")
    old_first_layer_weights = first_layer.weights.copy().flatten()

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

    new_first_layer_weights = first_layer.weights.copy().flatten()

    for i, values in enumerate(new_first_layer_weights):
        if values != 0:
            assert 2 * old_first_layer_weights[i] == new_first_layer_weights[i]


# Tests compressed training by compressing and decompressing weights between
# every batch update
def test_compressed_dragon_vector_training():
    acc = compressed_training(
        compression_scheme="dragon",
        compression_density=0.2,
        sample_population_size=100,
        hidden_dim=50,
        epochs=35,
    )
    assert acc[0]["categorical_accuracy"] >= ACCURACY_THRESHOLD
