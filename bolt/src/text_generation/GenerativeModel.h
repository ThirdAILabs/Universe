#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_set.hpp>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt_vector/src/BoltVector.h>
#include <licensing/src/CheckLicense.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace thirdai::bolt {

struct CandidateContext {
  std::vector<std::vector<uint32_t>> candidate_sequences;
  std::vector<uint32_t> prompt;
  std::vector<double> sequence_scores;

  CandidateContext(std::vector<std::vector<uint32_t>> candidate_sequences,
                   std::vector<double> sequence_scores,
                   std::vector<uint32_t> prompt)
      : candidate_sequences(std::move(candidate_sequences)),
        prompt(std::move(prompt)),
        sequence_scores(std::move(sequence_scores)) {}
};

class GenerativeBackend {
 public:
  virtual bolt::TensorPtr nextTokenProbs(
      const std::vector<std::vector<uint32_t>>& prompts,
      const std::vector<std::vector<std::vector<uint32_t>>>& tokens) = 0;

  virtual metrics::History train(const dataset::DataSourcePtr& train_data,
                                 float learning_rate, uint32_t epochs,
                                 size_t batch_size,
                                 const std::vector<std::string>& train_metrics,
                                 const dataset::DataSourcePtr& val_data,
                                 const std::vector<std::string>& val_metrics,
                                 std::optional<size_t> max_in_memory_batches,
                                 const DistributedCommPtr& comm) = 0;

  virtual ModelPtr getBoltModel() = 0;

  virtual ~GenerativeBackend() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

class GenerativeModel;

class BeamSearchDecoder {
 public:
  BeamSearchDecoder(std::shared_ptr<GenerativeModel> generator,
                    std::vector<std::vector<uint32_t>> prompts,
                    const std::vector<std::vector<uint32_t>>& input_tokens,
                    size_t prediction_chunk_size, size_t max_predictions,
                    size_t beam_width, std::optional<float> temperature)
      : _generator(std::move(generator)),
        _prediction_chunk_size(prediction_chunk_size),
        _max_predictions(max_predictions),
        _beam_width(beam_width),
        _temperature(temperature) {
    bool is_prompt_empty = prompts.empty();
    if (!is_prompt_empty && prompts.size() != input_tokens.size()) {
      throw std::runtime_error(
          "Inconsistent Size Detected. The 'prompts' container is not empty "
          "but its size does not matches 'input_tokens'. Ensure both of them "
          "have the same size.");
    }
    for (const auto& tokens : input_tokens) {
      _n_input_tokens.push_back(tokens.size());
    }
    for (uint32_t token_idx = 0; token_idx < input_tokens.size();
         token_idx += 1) {
      _candidate_contexts.push_back(CandidateContext(
          {input_tokens[token_idx]}, {0.0},
          is_prompt_empty ? std::vector<uint32_t>{} : prompts[token_idx]));
    }
  }

  std::optional<std::vector<std::vector<uint32_t>>> next();

  std::vector<std::vector<std::vector<uint32_t>>> getCandidateSequences() {
    std::vector<std::vector<std::vector<uint32_t>>> batched_candidate_sequences;
    batched_candidate_sequences.reserve(_candidate_contexts.size());
    for (auto& context : _candidate_contexts) {
      batched_candidate_sequences.push_back(context.candidate_sequences);
    }
    return batched_candidate_sequences;
  }

  std::vector<std::vector<uint32_t>> getPrompts() {
    std::vector<std::vector<uint32_t>> batched_prompts;
    batched_prompts.reserve(_candidate_contexts.size());
    for (auto& context : _candidate_contexts) {
      batched_prompts.push_back(context.prompt);
    }
    return batched_prompts;
  }

 private:
  void reduceProbsForRepeats(const std::vector<uint32_t>& sequence,
                             BoltVector& probs,
                             size_t exclude_repeats_range) const;

  static void applyTemperature(BoltVector& probs, float temperature);

  std::shared_ptr<GenerativeModel> _generator;

  std::vector<size_t> _n_input_tokens;
  const size_t _prediction_chunk_size;
  const size_t _max_predictions;
  const size_t _beam_width;
  const std::optional<float> _temperature;

  // This isues two seperate containers for the sequences and scores instead of
  // a std::vector<CandidateSequence> so that the sequences can be passed into
  // nextTokenProbs directly, instead of having to split apart the sequences and
  // scores.
  std::vector<CandidateContext> _candidate_contexts;
};

class GenerationStream {};

class GenerativeModel : public std::enable_shared_from_this<GenerativeModel> {
 private:
  GenerativeModel(std::shared_ptr<GenerativeBackend> model,
                  std::unordered_set<uint32_t> allowed_repeats,
                  std::unordered_set<uint32_t> punctuation_tokens,
                  float punctuation_repeat_threshold);

 public:
  static auto make(std::shared_ptr<GenerativeBackend> model,
                   std::unordered_set<uint32_t> allowed_repeats,
                   std::unordered_set<uint32_t> punctuation_tokens,
                   float punctuation_repeat_threshold) {
    return std::shared_ptr<GenerativeModel>(new GenerativeModel(
        std::move(model), std::move(allowed_repeats),
        std::move(punctuation_tokens), punctuation_repeat_threshold));
  }

  std::vector<uint32_t> generate(
      const std::vector<uint32_t>& input_tokens, std::vector<uint32_t> prompt,
      size_t max_predictions, size_t beam_width,
      std::optional<float> temperature = std::nullopt);

  std::vector<std::vector<uint32_t>> generateBatch(
      const std::vector<std::vector<uint32_t>>& input_tokens_batch,
      const std::vector<std::vector<uint32_t>>& prompt_batch,
      size_t max_predictions, size_t beam_width,
      std::optional<float> temperature);

  BeamSearchDecoder streamingGenerate(
      const std::vector<uint32_t>& input_tokens, std::vector<uint32_t> prompt,
      size_t prediction_chunk_size, size_t max_predictions, size_t beam_width,
      std::optional<float> temperature = std::nullopt);

  metrics::History train(
      const dataset::DataSourcePtr& train_data, float learning_rate,
      uint32_t epochs, size_t batch_size,
      const std::vector<std::string>& train_metrics = {},
      const dataset::DataSourcePtr& val_data = nullptr,
      const std::vector<std::string>& val_metrics = {},
      std::optional<size_t> max_in_memory_batches = std::nullopt,
      const DistributedCommPtr& comm = nullptr);

  const auto& model() const { return _model; }

  bool isAllowedRepeat(uint32_t token) const {
    return _allowed_repeats.count(token);
  }

  bool isPunct(uint32_t token) const {
    return _punctuation_tokens.count(token);
  }

  float punctuationRepeatThreshold() const {
    return _punctuation_repeat_threshold;
  }

  bolt::ModelPtr getBoltModel() { return _model->getBoltModel(); }

  void save(const std::string& filename) const;

  static std::shared_ptr<GenerativeModel> load(const std::string& filename);

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<GenerativeModel> load_stream(
      std::istream& input_stream);

 private:
  std::shared_ptr<GenerativeBackend> _model;

  std::unordered_set<uint32_t> _allowed_repeats;
  std::unordered_set<uint32_t> _punctuation_tokens;
  float _punctuation_repeat_threshold;

  GenerativeModel() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model, _allowed_repeats, _punctuation_tokens,
            _punctuation_repeat_threshold);
  }
};

}  // namespace thirdai::bolt