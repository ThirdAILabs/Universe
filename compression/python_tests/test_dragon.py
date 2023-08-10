import pytest

pytestmark = [pytest.mark.unit]

import numpy as np
from thirdai import bolt

from utils import compressed_training

INPUT_DIM = 100
HIDDEN_DIM = 100
OUTPUT_DIM = 100
LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8
TOLERANCE = 6
# A compressed vector is exposed as a char array
# and hence, it is not interpretable at Python end


def test_get_set_values_dragon_vector():
    original_array = np.random.randint(100, size=1000)
    compressed_array = bolt.compression.compress(
        original_array,
        compression_scheme="dragon",
        compression_density=0.2,
        seed_for_hashing=42,
        sample_population_size=10,
    )
    de_compressed_array = bolt.compression.decompress(compressed_array)
    value_mismatch = 0
    for i, value in enumerate(de_compressed_array):
        if value != 0:
            if original_array[i] != int(value):
                value_mismatch += 1
    assert value_mismatch <= TOLERANCE


def test_concat_values_dragon_vector():
    original_array = np.random.randint(100, size=1000)
    compressed_array = bolt.compression.compress(
        original_array,
        compression_scheme="dragon",
        compression_density=0.2,
        seed_for_hashing=42,
        sample_population_size=10,
    )
    concatenated_arr = bolt.compression.concat([compressed_array] * 2)
    de_compressed_array = bolt.compression.decompress(concatenated_arr)
    value_mismatch = 0
    for i, value in enumerate(de_compressed_array):
        if value != 0:
            if 2 * original_array[i] != int(value):
                value_mismatch += 1
    assert value_mismatch <= TOLERANCE


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
