from thirdai.bolt import MockHash, NWS
from collections import defaultdict


def test_nws(sparse):
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

    row_to_hash_to_weight = defaultdict(lambda: defaultdict(float))
    row_to_hash_to_count = defaultdict(lambda: defaultdict(float))

    for _, hashes, weight in inputs_hashes_weights:
        for row, row_hash in enumerate(hashes):
            row_to_hash_to_weight[row][row_hash] += weight
            row_to_hash_to_count[row][row_hash] += 1.0

    expectations = [
        sum(row_to_hash_to_weight[row][row_hash] for row, row_hash in enumerate(hashes))
        / sum(row_to_hash_to_count[row][row_hash] for row, row_hash in enumerate(hashes))
        for _, hashes, _ in inputs_hashes_weights
    ]

    print(expectations)

    hasher = MockHash([(inputs, hashes) for inputs, hashes, _ in inputs_hashes_weights])

    nws = NWS(hasher, sparse)
    inputs, _, weights = zip(*inputs_hashes_weights)
    nws.train(inputs, weights)
    for input_vec, expectation in zip(inputs, expectations):
        print(nws.predict([list(input_vec)])[0])
        assert abs(nws.predict([list(input_vec)])[0] - expectation) < 1e-6


test_nws(sparse=False)
test_nws(sparse=True)