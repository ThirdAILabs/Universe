#pragma once

#include "NerDyadicDataProcessor.h"
#include "NerTokenTagCounter.h"
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/ner/NerTokenFromStringArray.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace thirdai::data {

std::vector<std::string> cleanAndLowerCase(
    const std::vector<std::string>& tokens);

class NerTokenizerUnigram final : public Transformation {
 public:
  NerTokenizerUnigram(
      std::string tokens_column, std::string featurized_sentence_column,
      std::optional<std::string> target_column,
      std::optional<uint32_t> target_dim, uint32_t dyadic_num_intervals,
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      std::optional<FeatureEnhancementConfig> feature_enhancement_config,
      std::unordered_map<std::string, uint32_t> tag_to_label,
      std::unordered_set<std::string> ignored_tags,
      ner::TokenTagCounterPtr token_tag_counter);

  explicit NerTokenizerUnigram(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "ner_tokenization_unigram"; }

  std::string processToken(const std::vector<std::string>& tokens,
                           uint32_t index) const {
    return _processor.processToken(tokens, index, cleanAndLowerCase(tokens));
  }

  uint32_t findTagValueForString(const std::string& tag) const {
    if (_ignored_tags.count(tag)) {
      return 0;
    }

    if (_tag_to_label.count(tag)) {
      return _tag_to_label.at(tag);
    }

    throw std::out_of_range("String '" + tag +
                            "' not found in the specified tags list.");
  }

  const auto& processor() const { return _processor; }

  void setTokenTagCounter(ner::TokenTagCounterPtr token_tag_counter) {
    _token_tag_counter = std::move(token_tag_counter);
  }

 private:
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

  std::unordered_map<std::string, uint32_t> _tag_to_label;
  std::unordered_set<std::string> _ignored_tags;

  ner::TokenTagCounterPtr _token_tag_counter;
};

}  // namespace thirdai::data