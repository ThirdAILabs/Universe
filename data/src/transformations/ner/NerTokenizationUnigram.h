#pragma once

#include "UnigramDataProcessor.h"
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/NerTokenFromStringArray.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <string>

namespace thirdai::data {

class NerTokenizerUnigram final : public Transformation {
 public:
  NerTokenizerUnigram(
      std::string tokens_column, std::string featurized_sentence_column,
      std::optional<std::string> target_column, uint32_t fhr_dim,
      uint32_t dyadic_num_intervals,
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers);

  explicit NerTokenizerUnigram(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "ner_tokenization_unigram"; }

  std::string processToken(const std::vector<std::string>& tokens,
                           uint32_t index) const {
    return _processor.processToken(tokens, index);
  }

 private:
  std::string _tokens_column;
  std::string _featurized_sentence_column;
  std::optional<std::string> _target_column;

  SimpleDataProcessor _processor;
  std::string _featurized_tokens_indices_column =
      _featurized_sentence_column + "_tokens";
  std::string _featurized_tokens_values_column =
      _featurized_sentence_column + "_values";

  TransformationPtr _tokenizer_transformation;
};
}  // namespace thirdai::data