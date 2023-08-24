#include "BeamSearch.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <stdexcept>

namespace thirdai::search {

/**
 * C++ priority queues are structured such that the order the elements so that
 * the comparator returns true for any queue[i], queue[j] if i < j. The "top" of
 * the queue is the last element in this sequence. Since we are trying to
 * minimize the score we use the following comparator.
 */
struct Minimize {
  bool operator()(const Path& a, const Path& b) { return a.second < b.second; }
};

using CandidateQueue = std::priority_queue<Path, std::vector<Path>, Minimize>;

// Helper function to perform beam search on a single element of the batch.
std::vector<Path> beamSearch(const float* probabilities, uint32_t seq_len,
                             uint32_t output_dim,
                             const float* transition_matrix,
                             uint32_t beam_size) {
  // We keep a list of the top-k best scoring partial sequences that we update
  // at each step up to seq_len. This is what separates this approach from other
  // search algorithms, we limit the computations by only considering the best
  // beam_size possible sequences at any point, instead of all possible
  // sequences.
  std::vector<Path> candidate_sequences = {{{}, 0.0}};

  for (uint32_t seq_idx = 0; seq_idx < seq_len; seq_idx++) {
    // This will be ordered such that the worst scoring sequence is on the top
    // of the queue so we can easily check if a given sequence is better than at
    // least one of the candidates.
    CandidateQueue top_k;
    for (uint32_t i = 0; i < output_dim; i++) {
      for (const auto& seq : candidate_sequences) {
        float score =
            seq.second - std::log(probabilities[seq_idx * output_dim + i]);

        // Don't compute any transition probability for the first element of the
        // sequence.
        if (!seq.first.empty()) {
          uint32_t last_output = seq.first.back();
          score -= std::log(transition_matrix[last_output * output_dim + i]);
        }

        // If we have not found beam_size sequences yet, add the current
        // sequence to the set of candidates. If we have found beam_size
        // sequences then add the current sequence if its score is better than
        // the worst candidate.
        if (top_k.size() < beam_size || score < top_k.top().second) {
          std::vector<uint32_t> new_seq = seq.first;
          new_seq.push_back(i);

          top_k.emplace(std::move(new_seq), score);
          if (top_k.size() > beam_size) {
            top_k.pop();
          }
        }
      }
    }

    candidate_sequences.clear();
    candidate_sequences.reserve(top_k.size());

    // Update the candidate sequences with the new sequences that are one item
    // longer.
    while (!top_k.empty()) {
      candidate_sequences.push_back(top_k.top());
      top_k.pop();
    }
  }

  std::reverse(candidate_sequences.begin(), candidate_sequences.end());

  return candidate_sequences;
}

std::vector<std::vector<Path>> beamSearchBatch(
    const NumpyArray& probabilities, const NumpyArray& transition_matrix,
    uint32_t beam_size) {
  if (probabilities.ndim() != 3) {
    throw std::invalid_argument("probabilities should be 3D array.");
  }
  if (transition_matrix.ndim() != 2) {
    throw std::invalid_argument("Transition matrix should be 2D array.");
  }

  uint32_t batch_size = probabilities.shape(0);
  uint32_t seq_len = probabilities.shape(1);
  uint32_t output_dim = probabilities.shape(2);

  if (output_dim != transition_matrix.shape(0) ||
      output_dim != transition_matrix.shape(1)) {
    throw std::invalid_argument(
        "transition matrix shape does not match output dimension of "
        "probabilities.");
  }

  std::vector<std::vector<Path>> results(batch_size);

#pragma omp parallel for default(none)                                        \
    shared(batch_size, seq_len, output_dim, probabilities, transition_matrix, \
               beam_size, results)
  for (uint32_t i = 0; i < batch_size; i++) {
    results[i] = beamSearch(probabilities.data(i), seq_len, output_dim,
                            transition_matrix.data(), beam_size);
  }

  return results;
}

}  // namespace thirdai::search