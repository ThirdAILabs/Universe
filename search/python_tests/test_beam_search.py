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

    results = search.beam_search(probabilities, transition_matrix, beam_size=10)

    best_sequences = []
    for top_k in results:
        best_sequences.append(top_k[0][0])

    correct = np.argmax(probabilities, axis=2)

    assert np.array_equal(np.array(best_sequences), correct)


@pytest.mark.unit
def test_beam_search_differs_from_greedy_search():
    """
    This checks that the search algorithm correctly uses the transition matrix.
    The probability matrix looks like this (seq length = 3):

            P(0)   P(1)
     T0:    0.9    0.1
     T1:    0.45   0.55
     T2:    0.05   0.95

    And so a simple greedy approach will yield [0, 1, 1]. However when we consider
    the transition matrix:

              to: 0   to: 1
    from 0:    0.9     0.1
    from 1:    0.4     0.6

    We see that the best path is [0, 0, 1]. This is because if we look at the
    probability of [0, 0, 1] vs [0, 1, 1] we see:

    P(0, 1, 1) = 0.9 * 0.1 * 0.55 * 0.6 * 0.95
    P(0, 0, 1) = 0.9 * 0.9 * 0.45 * 0.1 * 0.95

    And so the difference is the term (0.1 * 0.55 * 0.6) vs (0.9 * 0.45 * 0.1).

    Note: we are looking at maximizing the probability whereas the beam search code
    tries to minimize the negative log sum of the probabilites. These are the same
    in terms of the path chosen.
    """
    probabilities = np.array([[[0.9, 0.1], [0.45, 0.55], [0.05, 0.95]]])

    transition_matrix = [[0.9, 0.1], [0.4, 0.6]]

    results = search.beam_search(probabilities, transition_matrix, beam_size=2)

    assert np.array_equal(results[0][0][0], np.array([0, 0, 1]))
