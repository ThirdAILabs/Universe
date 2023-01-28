#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace thirdai::search {

using NumpyArray = pybind11::array_t<float, pybind11::array::c_style |
                                                pybind11::array::forcecast>;

using Path = std::pair<std::vector<uint32_t>, float>;

/**
 * Performs beam search for sequence generation. Scores the sequences by
 * minimizing the negative log sum of the probabilities of each class at each
 * position in the sequence and the proabiblies of transitioning between each
 * pair of consecutive classes.
 *
 * Arguments:
 *  - An array of probabilities of shape (batch size, sequence length, N
 * outputs) which represent the outputs of a model predicting the probability
 * that the output is a given class for each element of the sequence.
 *  - A transition probability matrix of shape (N outputs, N outputs) where
 *    T[i,j] represents the probability of transitioning from class i to class j
 *    in a sequence. Finally it takes in
 *  - A value beam_size indicating the search buffer size to use and the number
 *    of candidate sequences to return.
 *
 * Returns:
 *  A list of the top beam_size sequences for each element in the batch. Where
 *  each sequence is a pair of the class id order for the sequence and the
 *  corresponding score.
 */
std::vector<std::vector<Path>> beamSearchBatch(
    const NumpyArray& probabilities, const NumpyArray& transition_matrix,
    uint32_t beam_size);

}  // namespace thirdai::search