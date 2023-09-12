#include "GenerativeModel.h"
#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <cmath>
#include <queue>
#include <utility>
#include <vector>

namespace thirdai::bolt {

GenerativeModel::GenerativeModel(
    std::shared_ptr<GenerativeBackend> model,
    std::unordered_set<uint32_t> allowed_repeats,
    std::unordered_set<uint32_t> punctuation_tokens)
    : _model(std::move(model)),
      _allowed_repeats(std::move(allowed_repeats)),
      _punctuation_tokens(std::move(punctuation_tokens)) {}

struct CandidateSequence {
  std::vector<uint32_t> sequence;
  double score;

  CandidateSequence(std::vector<uint32_t> _sequence, double _score)
      : sequence(std::move(_sequence)), score(_score) {}
};

struct MinimizeScore {
  bool operator()(const CandidateSequence& a,
                  const CandidateSequence& b) const {
    // We are taking the sum of the negative logs of the scores, thus best
    // sequences are the ones with the lowest scores. In the
    // std::priority_queue, the last element in the sorted sequence is the first
    // returned. Thus this ordering means that the highest scores will be at the
    // top of the queue.
    return a.score < b.score;
  }
};

using CandidateQueue =
    std::priority_queue<CandidateSequence, std::vector<CandidateSequence>,
                        MinimizeScore>;

std::vector<uint32_t> GenerativeModel::generate(
    const std::vector<uint32_t>& input_tokens, size_t n_predictions,
    size_t beam_width, std::optional<float> temperature) const {
  // This isues two seperate containers for the sequences and scores instead of
  // a std::vector<CandidateSequence> so that the sequences can be passed into
  // nextTokenProbs directly, instead of having to split apart the sequences and
  // scores.
  std::vector<std::vector<uint32_t>> candidate_sequences = {input_tokens};
  std::vector<double> sequence_scores = {0.0};

  for (size_t pred_idx = 0; pred_idx < n_predictions; pred_idx++) {
    auto next_token_probs = _model->nextTokenProbs(candidate_sequences);

    // This will be ordered such that the worst scoring sequence is on the top
    // of the queue so we can easily check if a given sequence is better than at
    // least one of the candidates already discovered.
    CandidateQueue candidates;

    for (size_t candidate = 0; candidate < candidate_sequences.size();
         candidate++) {
      BoltVector& token_probs = next_token_probs->getVector(candidate);
      reduceProbsForRepeats(candidate_sequences[candidate], token_probs,
                            n_predictions, temperature);

      auto top_tokens = token_probs.findKLargestActivations(beam_width);

      while (!top_tokens.empty()) {
        auto [prob, token] = top_tokens.top();
        top_tokens.pop();
        double score = sequence_scores[candidate] - std::log(prob);

        // If the candidates queue is not full, or if the new sequence has a
        // bettter score than the worst scoring sequence in the queue, then we
        // want to add the new sequence to the queue.
        if (candidates.size() < beam_width || candidates.top().score > score) {
          std::vector<uint32_t> new_sequence = candidate_sequences[candidate];
          new_sequence.push_back(token);

          candidates.emplace(std::move(new_sequence), score);

          if (candidates.size() > beam_width) {
            candidates.pop();
          }
        }
      }
    }

    candidate_sequences.clear();
    sequence_scores.clear();

    while (!candidates.empty()) {
      candidate_sequences.emplace_back(candidates.top().sequence);
      sequence_scores.push_back(candidates.top().score);
      candidates.pop();
    }
  }

  return {candidate_sequences.back().begin() + input_tokens.size(),
          candidate_sequences.back().end()};
}

void GenerativeModel::reduceProbsForRepeats(
    const std::vector<uint32_t>& sequence, BoltVector& probs,
    size_t n_predictions, std::optional<float> temperature) const {
  size_t start =
      sequence.size() < n_predictions ? 0 : sequence.size() - n_predictions;
  for (size_t i = start; i < sequence.size(); i++) {
    uint32_t token = sequence[i];

    if ((_punctuation_tokens.count(token) && probs.activations[token] < 0.8) ||
        !_allowed_repeats.count(token)) {
      probs.activations[token] = 0.0;
    }
  }

  if (temperature) {
    float total = 0.0;
    for (size_t i = 0; i < probs.len; i++) {
      float score = std::exp(probs.activations[i] / *temperature);
      probs.activations[i] = score;
      total += score;
    }

    for (size_t i = 0; i < probs.len; i++) {
      probs.activations[i] = probs.activations[i] / total;
    }
  }
}

metrics::History GenerativeModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const DistributedCommPtr& comm) {
  return _model->train(train_data, learning_rate, epochs, batch_size,
                       train_metrics, val_data, val_metrics, comm);
}

void GenerativeModel::save(const std::string& filename) const {
  auto output_stream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<GenerativeModel> GenerativeModel::load(
    const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);

  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<GenerativeModel> deserialize_into(new GenerativeModel());
  iarchive(*deserialize_into);

  return deserialize_into;
}

}  // namespace thirdai::bolt