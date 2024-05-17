#pragma once

#include "UnigramDataProcessor.h"
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/ner/NerTokenFromStringArray.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <stdexcept>
#include <string>

namespace thirdai::data {

class NerTokenizerUnigram final : public Transformation {
 public:
  NerTokenizerUnigram(
      std::string tokens_column, std::string featurized_sentence_column,
      std::optional<std::string> target_column,
      std::optional<uint32_t> target_dim, uint32_t fhr_dim,
      uint32_t dyadic_num_intervals,
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      std::optional<std::unordered_map<std::string, uint32_t>> tag_to_label =
          std::nullopt);

  explicit NerTokenizerUnigram(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "ner_tokenization_unigram"; }

  std::string processToken(const std::vector<std::string>& tokens,
                           uint32_t index) const {
    return _processor.processToken(tokens, index);
  }

  std::string getFeaturizedIndicesColumn() const {
    return _featurized_tokens_indices_column;
  }

  uint32_t findTagValueForString(const std::string& tag) const {
    if (!_tag_to_label.has_value()) {
      throw std::logic_error("Tag to Label is None");
    }
    auto tag_map = _tag_to_label.value();
    if (tag_map.count(tag)) {
      return tag_map.at(tag);
    }

    throw std::out_of_range("String not found in the label map: " + tag);
  }

 private:
  /*
   * _tokens_column : the column containing the string tokens
   * _target_column : the column containing the target tags
   * _target_dim : the number of total different labels
   * _featurized_tokens_indices_column : this column contains the tokens after
   * tokenizing the _featurized_sentence_column
   * _tokenization_transformation : the transformation used to tokenize the
   * featurized sentences
   */
  std::string _tokens_column;
  std::string _featurized_sentence_column;
  std::optional<std::string> _target_column;
  std::optional<uint32_t> _target_dim;

  NerDyadicDataProcessor _processor;
  std::string _featurized_tokens_indices_column = "featurized_tokens_indices_column";

  // TODO(Shubh) : Add support for depuplicating the tokens by using indices and
  // values pair.

  TransformationPtr _tokenizer_transformation;
  std::optional<std::unordered_map<std::string, uint32_t>> _tag_to_label;
};
}  // namespace thirdai::data