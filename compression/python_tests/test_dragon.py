import pytest

pytestmark = [pytest.mark.unit]

import numpy as np
from thirdai import bolt

from utils import build_simple_hidden_layer_model, compressed_training

INPUT_DIM = 100
HIDDEN_DIM = 100
OUTPUT_DIM = 100
LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8
TOLERANCE = 6
# A compressed vector is exposed as a char array
# and hence, it is not interpretable at Python end


def test_get_set_values_dragon_vector():
    model = build_simple_hidden_layer_model(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM
    )
    model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

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
    # set is parallel in dragon and is not thread safe and there
    # might be a race condition when two indices go the sum bucket
    # in the dragon vector and are picked up by two threads at the same time.
    # that is indices[i], values[j] is possible if hash(i) == hash(j) and are
    # in a race condition.

    number_weights_mismatch = 0
    for i, values in enumerate(new_first_layer_weights):
        if values != 0:
            if old_first_layer_weights[i] != new_first_layer_weights[i]:
                number_weights_mismatch += 1
    number_biases_mismatch = 0
    for i, values in enumerate(new_first_layer_biases):
        if values != 0:
            if old_first_layer_biases[i] != new_first_layer_biases[i]:
                number_biases_mismatch += 1

    assert number_weights_mismatch <= TOLERANCE and number_biases_mismatch <= TOLERANCE


def test_concat_values_dragon_vector():
    model = build_simple_hidden_layer_model(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM
    )
    model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

    first_layer = model.get_layer("fc_1")
    old_first_layer_weights = first_layer.weights.copy().flatten()

    # getting the compressed gradients
    compressed_weights = first_layer.weights.compress(
        compression_scheme="dragon",
        compression_density=0.2,
        seed_for_hashing=1,
        sample_population_size=50,
    )
    concatenated_weights = bolt.nn.ParameterReference.concat([compressed_weights] * 2)
    first_layer.weights.set(concatenated_weights)

    new_first_layer_weights = first_layer.weights.copy().flatten()
    number_weights_mismatch = 0
    for i, values in enumerate(new_first_layer_weights):
        if values != 0:
            if 2 * old_first_layer_weights[i] != new_first_layer_weights[i]:
                number_weights_mismatch += 1
    assert number_weights_mismatch <= TOLERANCE


# Tests compressed training by compressing and decompressing weights between
# every batch update
def test_compressed_dragon_vector_training():
    acc = compressed_training(
        compression_scheme="dragon",
        compression_density=0.2,
        sample_population_size=100,
        epochs=50,
    )
    assert acc >= ACCURACY_THRESHOLD
