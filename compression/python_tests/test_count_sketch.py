import pytest

pytestmark = [pytest.mark.unit]

from utils import compressed_training

LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8


# Tests compressed training by compressing and decompressing weights between
# every batch update
def test_compressed_count_sketch_training():
    num_sketches = 3
    compression_density = 0.3
    num_epochs = 50
    n_classes = 10

    acc = compressed_training(
        compression_scheme="count_sketch",
        compression_density=compression_density,
        sample_population_size=num_sketches,
        epochs=num_epochs,
        n_classes=n_classes,
    )
    assert acc >= ACCURACY_THRESHOLD
