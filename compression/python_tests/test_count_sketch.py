import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]

import numpy as np
from thirdai import bolt, dataset

from utils import (
    compressed_training,
)

LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8

# Tests compressed training by compressing and decompressing weights between
# every batch update
def test_compressed_count_sketch_training():
    num_sketches = 1
    compression_density = 0.25
    num_epochs = 35

    acc = compressed_training(
        compression_scheme="count_sketch",
        compression_density=compression_density,
        sample_population_size=num_sketches,
        hidden_dim=50,
        epochs=num_epochs,
    )
    assert acc[0]["categorical_accuracy"] >= ACCURACY_THRESHOLD
