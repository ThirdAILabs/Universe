#pragma once

#include "NerDyadicDataProcessor.h"
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/ner/NerTokenFromStringArray.h>
#include <data/src/transformations/ner/utils/TokenTagCounter.h>
#include <data/src/transformations/ner/utils/utils.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <stdexcept>
#include <string>

namespace thirdai::data {

class NerTokenizerUnigram final : public Transformation {
 public:
  NerTokenizerUnigram(
      std::string tokens_column, std::string featurized_sentence_column,
      std::optional<std::string> target_column,
      std::optional<uint32_t> target_dim, uint32_t dyadic_num_intervals,
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      std::optional<FeatureEnhancementConfig> feature_enhancement_config,
      std::optional<std::unordered_map<std::string, uint32_t>> tag_to_label =
          std::nullopt,
      ner::TokenTagCounterPtr token_tag_counter = nullptr);

  explicit NerTokenizerUnigram(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "ner_tokenization_unigram"; }

  std::string processToken(const std::vector<std::string>& tokens,
                           uint32_t index) const {
    return _processor.processToken(tokens, index,
                                   ner::utils::cleanAndLowerCase(tokens));
  }

  uint32_t findTagValueForString(const std::string& tag) const {
    if (!_tag_to_label.has_value()) {
      throw std::logic_error("Tag to Label is None");
    }

    auto tag_map = _tag_to_label.value();
    if (tag_map.count(tag)) {
      return tag_map.at(tag);
    }

    throw std::out_of_range("String '" + tag +
                            "' not found in the specified tags list.");
  }

  const auto& processor() const { return _processor; }

  void setTokenTagCounter(ner::TokenTagCounterPtr token_tag_counter) {
    _token_tag_counter = std::move(token_tag_counter);
  }

 private:
  void updateTokenTagCounter(
      const ArrayColumnBasePtr<std::string>& tokens,
      const ArrayColumnBasePtr<std::string>& tags) const {
    if (_token_tag_counter != nullptr && _target_column.has_value()) {
      for (size_t i = 0; i < tokens->numRows(); ++i) {
        std::vector<std::string> row_token_vectors = tokens->row(i).toVector();
        auto lower_cased_tokens =
            ner::utils::cleanAndLowerCase(row_token_vectors);

        for (size_t token_index = 0; token_index < row_token_vectors.size();
             ++token_index) {
          _token_tag_counter->addTokenTag(lower_cased_tokens[token_index],
                                          tags->row(i)[token_index]);
        }
      }
    }
  }
  /*
   * _tokens_column : the column containing the string tokens
   * _target_column : the column containing the target tags
   * _target_dim : the number of total different labels
   */
  std::string _tokens_column;
  std::string _featurized_sentence_column;
  std::optional<std::string> _target_column;
  std::optional<uint32_t> _target_dim;

  NerDyadicDataProcessor _processor;

  std::optional<std::unordered_map<std::string, uint32_t>> _tag_to_label;

  ner::TokenTagCounterPtr _token_tag_counter;
};

}  // namespace thirdai::data