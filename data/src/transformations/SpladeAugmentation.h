#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>
#include <regex>

namespace thirdai::data {

struct SpladeConfig {
  SpladeConfig(const std::string& model_checkpoint,
               const std::string& tokenizer_vocab,
               std::optional<size_t> n_augmented_tokens,
               std::optional<float> augmentation_frac,
               bool filter_tokens = true, size_t batch_size = 4096,
               bool lowercase = true,
               std::optional<uint32_t> strong_sample_override = 7);

  bolt::ModelPtr model;
  dataset::WordpieceTokenizerPtr tokenizer;
  std::optional<size_t> n_augmented_tokens;
  std::optional<float> augmentation_frac;
  bool filter_tokens;
  size_t batch_size = 4096;
  std::optional<uint32_t> strong_sample_override;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<SpladeConfig> load_stream(std::istream& input_stream);

 private:
  // Rather than saving the model and tokenizer, this object just stores the
  // paths and recreates them on load, this is to avoid the cost of serializtion
  // the model for NDB checkpointing, which is really the only case in which
  // this should be serialized.
  std::string _model_checkpoint;
  std::string _tokenizer_vocab;
  bool _lowercase = true;
};

class SpladeAugmentation final : public Transformation {
 public:
  SpladeAugmentation(std::string input_column,
                    std::string output_indices_column,
                    std::string output_values_column,
                    const SpladeConfig& config);

  SpladeAugmentation(std::string input_column, std::string output_indices_column,
                     std::string output_values_column,
                     bolt::ModelPtr model,
                     dataset::WordpieceTokenizerPtr tokenizer,
                     std::optional<size_t> n_augmented_tokens,
                     std::optional<float> augmentation_frac, bool filter_tokens,
                     size_t batch_size);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

 private:
  std::pair<std::vector<size_t>, std::vector<float>> decodeTopTokens(const BoltVector& vec, size_t k) const;

  inline size_t tokensToAdd(size_t seq_len) const {
    if (_n_augmented_tokens) {
      return _n_augmented_tokens.value();
    }
    return seq_len * _augmentation_frac.value();
  }

  std::string _input_column;
  std::string _output_indices_column;
  std::string _output_values_column;

  bolt::ModelPtr _model;
  dataset::WordpieceTokenizerPtr _tokenizer;

  std::optional<size_t> _n_augmented_tokens;
  std::optional<float> _augmentation_frac;
  bool _filter_tokens;
  size_t _batch_size = 4096;

  const std::regex _allowed_tokens = std::regex(R"([a-zA-Z]{3,})");
};

}  // namespace thirdai::data