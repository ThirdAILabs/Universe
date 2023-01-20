#include "BeamSearch.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <queue>
#include <stdexcept>

namespace thirdai::search {

struct Minimize {
  bool operator()(const SeqResult& a, const SeqResult& b) {
    return a.second < b.second;
  }
};

using CandidateQueue =
    std::priority_queue<SeqResult, std::vector<SeqResult>, Minimize>;

std::vector<SeqResult> beamSearch(const float* probabilies, uint32_t seq_len,
                                  uint32_t output_dim,
                                  const NumpyArray& transistion_matrix,
                                  uint32_t k) {
  std::vector<SeqResult> candidate_sequences = {{{}, 0.0}};

  for (uint32_t seq_idx = 0; seq_idx < seq_len; seq_idx++) {
    CandidateQueue top_k;
    for (uint32_t i = 0; i < output_dim; i++) {
      for (const auto& seq : candidate_sequences) {
        float probability = std::log(probabilies[seq_idx * output_dim + i]);

        if (!seq.first.empty()) {
          probability += std::log(transistion_matrix.at(seq.first.back(), i));
        }

        float score = seq.second - probability;

        if (top_k.size() < k || score < top_k.top().second) {
          std::vector<uint32_t> new_seq = seq.first;
          new_seq.push_back(i);

          top_k.emplace(std::move(new_seq), score);
          if (top_k.size() > k) {
            top_k.pop();
          }
        }
      }
    }

    candidate_sequences.clear();
    candidate_sequences.reserve(top_k.size());

    while (!top_k.empty()) {
      candidate_sequences.push_back(top_k.top());
      top_k.pop();
    }
  }

  return candidate_sequences;
}

std::vector<std::vector<SeqResult>> beamSearchBatch(
    const NumpyArray& probabilities, const NumpyArray& transition_matrix,
    uint32_t k) {
  if (probabilities.ndim() != 3) {
    throw std::invalid_argument("Probabilies should be 3D array.");
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
        "Transistion matrix shape does not match output dimension of "
        "probabilities.");
  }

  std::vector<std::vector<SeqResult>> results(batch_size);

#pragma omp parallel for default(none)                                        \
    shared(batch_size, seq_len, output_dim, probabilities, transition_matrix, \
           k, results)
  for (uint32_t i = 0; i < batch_size; i++) {
    results[i] = beamSearch(probabilities.data(i), seq_len, output_dim,
                            transition_matrix, k);
  }

  return results;
}

}  // namespace thirdai::search