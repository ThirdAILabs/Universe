import pytest
from thirdai import dataset
import random
import numpy as np


@pytest.mark.unit
def test_dataset():
    num_samples = 4
    num_batches = 5
    seed = 42
    generator = random.Random(seed)

    # Max length of a sample.
    MAX_LENGTH = 30

    batches = []
    for batch_id in range(num_batches):
        samples = []
        for sample_id in range(num_samples):
            # Create samples
            length = generator.randint(1, MAX_LENGTH)
            candidate_indices = list(range(0, MAX_LENGTH - 1))
            generator.shuffle(candidate_indices)
            indices = candidate_indices[:length]
            values = [generator.random() for _ in candidate_indices]
            values = np.array(values, dtype=np.float32)
            print(indices, values)
            # indices = [1]
            # values = [2.0]
            sample = dataset.make_sparse_vector(indices=indices, values=values)
            print(indices, values)
            samples.append(sample)

        # Create batch from samples
        batch = dataset.BoltBatch(samples)
        batches.append(batch)

    # Create dataset
    _dataset = dataset.BoltDataset(batches)


if __name__ == "__main__":
    test_dataset()
