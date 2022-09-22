import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]

import numpy as np
from thirdai import bolt, dataset

from utils import (
    compressed_training,
)

LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8

# Compress the weight gradients of the model using count sketches, and then
# train the model using compressed gradients
def test_compressed_count_sketch_training():
    num_sketches = 1
    compression_density = 0.2
    num_epochs = 30

    acc = compressed_training(
        compression_scheme="count_sketch",
        compression_density=compression_density,
        sample_population_size=num_sketches,
        hidden_dim=50,
        epochs=num_epochs,
    )
    assert acc[0]["categorical_accuracy"] >= ACCURACY_THRESHOLD
