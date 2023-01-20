import numpy as np
import pytest
from thirdai import search


@pytest.mark.unit
def test_beam_search_no_transition_matrix():
    """
    Because the transition probablity matrix is all ones, this equivalent to the
    argmax because there is no benefit from choosing anything other than the best
    class at each point in the sequence.
    """
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


@pytest.mark.unit
def test_beam_search_differs_from_gready_search():
    """
    This checks that the search algorithm correctly uses the transition matrix.
    The probability matrix looks like this (seq length = 3):

        0.9   0.1
        0.45  0.55
        0.05  0.95

    And so a simple greedy approach will yield [0, 1, 1]. However when we consider
    the transition matrix:

        0.9  0.1
        0.4  0.6

    We see that the best path is [0, 0, 1]
    """
    probabilities = np.array([[[0.9, 0.1], [0.45, 0.55], [0.05, 0.95]]])

    transition_matrix = [[0.9, 0.1], [0.4, 0.6]]

    results = search.beam_search(probabilities, transition_matrix, k=2)

    assert np.array_equal(results[0][-1][0], np.array([0, 0, 1]))
