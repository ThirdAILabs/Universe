#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>

namespace thirdai::data {

struct SpladeConfig {
  SpladeConfig(std::string model_checkpoint, std::string tokenizer_vocab,
               std::optional<size_t> n_augmented_tokens,
               std::optional<float> augmentation_frac, size_t batch_size = 4096,
               bool lowercase = true)
      : model_checkpoint(std::move(model_checkpoint)),
        tokenizer_vocab(std::move(tokenizer_vocab)),
        n_augmented_tokens(n_augmented_tokens),
        augmentation_frac(augmentation_frac),
        batch_size(batch_size),
        lowercase(lowercase) {}

  std::string model_checkpoint;
  std::string tokenizer_vocab;
  std::optional<size_t> n_augmented_tokens;
  std::optional<float> augmentation_frac;
  size_t batch_size = 4096;
  bool lowercase = true;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<SpladeConfig> load_stream(std::istream& input_stream);
};

class SpladeAugmentation final : public Transformation {
 public:
  SpladeAugmentation(std::string input_column, std::string output_column,
                     const SpladeConfig& config);

  SpladeAugmentation(std::string input_column, std::string output_column,
                     bolt::ModelPtr model,
                     dataset::WordpieceTokenizerPtr tokenizer,
                     std::optional<size_t> n_augmented_tokens,
                     std::optional<float> augmentation_frac, size_t batch_size);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

 private:
  std::string decodeTopTokens(const BoltVector& vec, size_t k) const;

  inline size_t tokensToAdd(size_t seq_len) const {
    if (_n_augmented_tokens) {
      return _n_augmented_tokens.value();
    }
    return seq_len * _augmentation_frac.value();
  }

  std::string _input_column;
  std::string _output_column;

  bolt::ModelPtr _model;
  dataset::WordpieceTokenizerPtr _tokenizer;

  std::optional<size_t> _n_augmented_tokens;
  std::optional<float> _augmentation_frac;
  size_t _batch_size = 4096;
};

}  // namespace thirdai::data