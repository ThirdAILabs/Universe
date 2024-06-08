from thirdai.bolt import MockHash, RACE
from collections import defaultdict


def test_dense_race():
    inputs_hashes_weights = [
        ([0.0, 1.0, 2.0], [2, 1, 0], 0.3),
        ([1.0, 2.0, 3.0], [0, 0, 1], 0.6),
        ([2.0, 3.0, 4.0], [2, 0, 1], 0.9),
        ([3.0, 4.0, 5.0], [1, 0, 2], 1.2),
        ([4.0, 5.0, 6.0], [0, 1, 1], 1.5),
        ([5.0, 6.0, 7.0], [0, 2, 1], 1.8),
        ([6.0, 7.0, 8.0], [1, 1, 2], 2.1),
        ([7.0, 8.0, 9.0], [0, 0, 1], 2.4),
    ]

    row_to_hash_to_count = defaultdict(lambda: defaultdict(float))

    for _, hashes, weight in inputs_hashes_weights:
        for row, row_hash in enumerate(hashes):
            row_to_hash_to_count[row][row_hash] += weight

    expectations = [
        sum(row_to_hash_to_count[row][row_hash] for row, row_hash in enumerate(hashes))
        / 3
        for _, hashes, _ in inputs_hashes_weights
    ]

    hasher = MockHash([(inputs, hashes) for inputs, hashes, _ in inputs_hashes_weights])

    race = RACE(hasher)
    inputs, _, weights = zip(*inputs_hashes_weights)
    for input_vec, weight in zip(inputs, weights):
        race.update(list(input_vec), weight)
    for input_vec, expectation in zip(inputs, expectations):
        assert abs(race.query(list(input_vec)) - expectation) < 1e-6


def test_sparse_race():
    pass
