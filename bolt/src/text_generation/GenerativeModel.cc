#include "GenerativeModel.h"
#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <cmath>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

namespace thirdai::bolt {

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

std::optional<std::vector<uint32_t>> BeamSearchDecoder::next() {
  size_t n_predictions =
      std::min(_prediction_chunk_size, _n_input_tokens + _max_predictions -
                                           _candidate_sequences.front().size());

  if (n_predictions == 0) {
    return std::nullopt;
  }

  for (size_t pred_idx = 0; pred_idx < n_predictions; pred_idx++) {
    auto next_token_probs =
        _generator->model()->nextTokenProbs(_candidate_sequences);

    // This will be ordered such that the worst scoring sequence is on the top
    // of the queue so we can easily check if a given sequence is better than at
    // least one of the candidates already discovered.
    CandidateQueue candidates;

    for (size_t candidate = 0; candidate < _candidate_sequences.size();
         candidate++) {
      BoltVector& token_probs = next_token_probs->getVector(candidate);
      reduceProbsForRepeats(_candidate_sequences[candidate], token_probs,
                            _max_predictions);

      if (_temperature) {
        applyTemperature(token_probs, *_temperature);
      }

      auto top_tokens = token_probs.findKLargestActivations(_beam_width);

      while (!top_tokens.empty()) {
        auto [prob, token] = top_tokens.top();
        top_tokens.pop();
        double score = _sequence_scores[candidate] - std::log(prob);

        // If the candidates queue is not full, or if the new sequence has a
        // better score than the worst scoring sequence in the queue, then we
        // want to add the new sequence to the queue.
        if (candidates.size() < _beam_width || candidates.top().score > score) {
          std::vector<uint32_t> new_sequence = _candidate_sequences[candidate];
          new_sequence.push_back(token);

          candidates.emplace(std::move(new_sequence), score);

          if (candidates.size() > _beam_width) {
            candidates.pop();
          }
        }
      }
    }

    _candidate_sequences.clear();
    _sequence_scores.clear();

    while (!candidates.empty()) {
      _candidate_sequences.emplace_back(candidates.top().sequence);
      _sequence_scores.push_back(candidates.top().score);
      candidates.pop();
    }
  }

  std::vector<uint32_t> tokens = {
      _candidate_sequences.back().begin() + _n_input_tokens,
      _candidate_sequences.back().end()};

  return tokens;
}

void BeamSearchDecoder::reduceProbsForRepeats(
    const std::vector<uint32_t>& sequence, BoltVector& probs,
    size_t exclude_repeats_range) const {
  size_t start = sequence.size() < exclude_repeats_range
                     ? 0
                     : sequence.size() - exclude_repeats_range;

  for (size_t i = start; i < sequence.size(); i++) {
    uint32_t token = sequence[i];

    if (_generator->isPunct(token)) {
      if (probs.activations[token] < _generator->punctuationRepeatThreshold()) {
        probs.activations[token] = 0.0;
      }
    } else if (!_generator->isAllowedRepeat(token)) {
      probs.activations[token] = 0.0;
    }
  }
}

void BeamSearchDecoder::applyTemperature(BoltVector& probs, float temperature) {
  float total = 0.0;
  for (size_t i = 0; i < probs.len; i++) {
    float score = std::exp(std::log(probs.activations[i]) / temperature);
    probs.activations[i] = score;
    total += score;
  }

  for (size_t i = 0; i < probs.len; i++) {
    probs.activations[i] = probs.activations[i] / total;
  }
}

GenerativeModel::GenerativeModel(
    std::shared_ptr<GenerativeBackend> model,
    std::unordered_set<uint32_t> allowed_repeats,
    std::unordered_set<uint32_t> punctuation_tokens,
    float punctuation_repeat_threshold)
    : _model(std::move(model)),
      _allowed_repeats(std::move(allowed_repeats)),
      _punctuation_tokens(std::move(punctuation_tokens)),
      _punctuation_repeat_threshold(punctuation_repeat_threshold) {}

std::vector<uint32_t> GenerativeModel::generate(
    const std::vector<uint32_t>& input_tokens, size_t n_predictions,
    size_t beam_width, std::optional<float> temperature) {
  BeamSearchDecoder decoder(shared_from_this(), input_tokens, n_predictions,
                            n_predictions, beam_width, temperature);

  return decoder.next().value_or(std::vector<uint32_t>{});
}

BeamSearchDecoder GenerativeModel::streamingGeneration(
    const std::vector<uint32_t>& input_tokens, size_t prediction_chunk_size,
    size_t max_predictions, size_t beam_width,
    std::optional<float> temperature) {
  return BeamSearchDecoder(shared_from_this(), input_tokens,
                           prediction_chunk_size, max_predictions, beam_width,
                           temperature);
}

metrics::History GenerativeModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const DistributedCommPtr& comm) {
  licensing::entitlements().verifyFullAccess();

  return _model->train(train_data, learning_rate, epochs, batch_size,
                       train_metrics, val_data, val_metrics, comm);
}

void GenerativeModel::save(const std::string& filename) const {
  auto output_stream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

void GenerativeModel::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<GenerativeModel> GenerativeModel::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<GenerativeModel> deserialize_into(new GenerativeModel());
  iarchive(*deserialize_into);

  return deserialize_into;
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