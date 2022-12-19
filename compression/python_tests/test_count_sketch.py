import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]

import numpy as np
from thirdai import bolt, dataset

from utils import build_simple_hidden_layer_model, compressed_training

LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8

INPUT_DIM = 10
HIDDEN_DIM = 10
OUTPUT_DIM = 10

# The estimation of a CountSketch does not change on adding/extending it with
# an identical CountSketch
def test_concat_values_count_sketch():
    model = build_simple_hidden_layer_model(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM
    )
    model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

    first_layer = model.get_layer("fc_1")

    # getting the compressed gradients
    compressed_weights = first_layer.weights.compress(
        compression_scheme="count_sketch",
        compression_density=0.3,
        seed_for_hashing=1,
        sample_population_size=1,
    )
    first_layer.weights.set(compressed_weights)

    old_first_layer_weights = first_layer.weights.copy().flatten()
    concatenated_weights = bolt.nn.ParameterReference.concat([compressed_weights] * 2)
    first_layer.weights.set(concatenated_weights)

    new_first_layer_weights = first_layer.weights.copy().flatten()

    for i, values in enumerate(new_first_layer_weights):
        if values != 0:
            assert old_first_layer_weights[i] == new_first_layer_weights[i]


def test_add_count_sketch():
    model = build_simple_hidden_layer_model(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM
    )
    model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

    first_layer = model.get_layer("fc_1")

    compressed_weights = first_layer.weights.compress(
        compression_scheme="count_sketch",
        compression_density=0.3,
        seed_for_hashing=1,
        sample_population_size=1,
    )
    first_layer.weights.set(compressed_weights)

    old_first_layer_weights = first_layer.weights.copy().flatten()
    aggregated_weights = bolt.nn.ParameterReference.add([compressed_weights])
    first_layer.weights.set(aggregated_weights)

    new_first_layer_weights = first_layer.weights.copy().flatten()
    for i, values in enumerate(new_first_layer_weights):
        if values != 0:
            assert old_first_layer_weights[i] == new_first_layer_weights[i]


# Tests compressed training by compressing and decompressing weights between
# every batch update
def test_compressed_count_sketch_training():
    num_sketches = 3
    compression_density = 0.3
    num_epochs = 50

    acc = compressed_training(
        compression_scheme="count_sketch",
        compression_density=compression_density,
        sample_population_size=num_sketches,
        hidden_dim=60,
        epochs=num_epochs,
    )
    assert acc[0]["categorical_accuracy"] >= ACCURACY_THRESHOLD
