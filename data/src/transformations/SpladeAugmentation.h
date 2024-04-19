#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/WordpieceTokenizer.h>
#include <regex>

namespace thirdai::data {

class SpladeAugmentation final : public Transformation {
 public:
  SpladeAugmentation(std::string input_column, std::string output_column,
                     bolt::ModelPtr model,
                     dataset::WordpieceTokenizerPtr tokenizer,
                     std::optional<size_t> n_augmented_tokens,
                     std::optional<float> augmentation_frac, bool filter_tokens,
                     size_t batch_size, std::optional<size_t> token_offset=std::nullopt);

  explicit SpladeAugmentation(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "splade_augmentation"; }

  bool hasTokenOffset() { return _token_offset.has_value();}

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
  bool _filter_tokens;
  size_t _batch_size = 4096;

  std::optional<size_t> _token_offset;

  const std::regex _allowed_tokens = std::regex(R"([a-zA-Z]{3,})");
};

}  // namespace thirdai::data