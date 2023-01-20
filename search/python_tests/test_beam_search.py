import numpy as np
import pytest
from thirdai import search


@pytest.mark.unit
def test_beam_search_no_transition_matrix():
    output_dim = 100
    probabilities = np.random.rand(10, 20, output_dim)

    transition_matrix = np.ones(shape=(output_dim, output_dim), dtype=np.float32)

    results = search.beam_search(probabilities, transition_matrix, k=10)

    best_sequences = []
    for top_k in results:
        best_score = float("inf")
        best_seq = None
        for seq, score in top_k:
            if score < best_score:
                best_seq = seq
        best_sequences.append(best_seq)

    correct = np.argmax(probabilities, axis=2)

    assert np.array_equal(np.array(best_sequences), correct)
