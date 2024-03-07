#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>

namespace thirdai::data {

struct SpladeConfig {
  SpladeConfig(std::string model_checkpoint, std::string tokenizer_vocab,
               size_t n_augmented_tokens, bool lowercase = true)
      : model_checkpoint(std::move(model_checkpoint)),
        tokenizer_vocab(std::move(tokenizer_vocab)),
        n_augmented_tokens(n_augmented_tokens),
        lowercase(lowercase) {}

  std::string model_checkpoint;
  std::string tokenizer_vocab;
  size_t n_augmented_tokens;
  bool lowercase = true;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<SpladeConfig> load_stream(std::istream& input_stream);
};

class SpladeAugmentation final : public Transformation {
 public:
  SpladeAugmentation(std::string input_column, std::string output_column,
                     const SpladeConfig& config)
      : SpladeAugmentation(std::move(input_column), std::move(output_column),
                           config.model_checkpoint, config.tokenizer_vocab,
                           config.n_augmented_tokens, config.lowercase) {}

  SpladeAugmentation(std::string input_column, std::string output_column,
                     const std::string& model_checkpoint,
                     const std::string& tokenizer_vocab,
                     size_t n_augmented_tokens, bool lowercase = true);

  SpladeAugmentation(std::string input_column, std::string output_column,
                     bolt::ModelPtr model,
                     dataset::WordpieceTokenizerPtr tokenizer,
                     size_t n_augmented_tokens);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

 private:
  std::string decodeTopTokens(const BoltVector& vec, size_t k) const;

  std::string _input_column;
  std::string _output_column;

  bolt::ModelPtr _model;
  dataset::WordpieceTokenizerPtr _tokenizer;

  size_t _n_augmented_tokens;
};

}  // namespace thirdai::data