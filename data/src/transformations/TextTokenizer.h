#pragma once

#include <cereal/access.hpp>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/utils/TokenEncoding.h>

namespace thirdai::data {

class TextTokenizer final : public Transformation {
 public:
  TextTokenizer(
      std::string input_column, std::string output_indices,
      std::optional<std::string> output_values,
      dataset::TextTokenizerPtr tokenizer, dataset::TextEncoderPtr encoder,
      bool lowercase = false,
      size_t dim = dataset::token_encoding::DEFAULT_TEXT_ENCODING_DIM);

  explicit TextTokenizer(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "text_tokenizer"; }

 private:
  static std::pair<std::vector<uint32_t>, std::vector<float>>
  deduplicateIndices(std::vector<uint32_t>&& tokens);

  std::string _input_column, _output_indices;
  std::optional<std::string> _output_values;

  dataset::TextTokenizerPtr _tokenizer;
  dataset::TextEncoderPtr _encoder;

  bool _lowercase;
  bool _clean_text;  // Placeholder to avoid compatability issue, unused now.
  size_t _dim;

  TextTokenizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using TextTokenizerPtr = std::shared_ptr<TextTokenizer>;

}  // namespace thirdai::data