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
#include <data/src/transformations/ner/utils/TagTracker.h>
#include <data/src/transformations/ner/utils/utils.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::data {

class NerTokenizerUnigram final : public Transformation {
 public:
  NerTokenizerUnigram(
      std::string tokens_column, std::string featurized_sentence_column,
      std::optional<std::string> target_column, uint32_t dyadic_num_intervals,
      std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
      std::optional<FeatureEnhancementConfig> feature_enhancement_config,
      ner::utils::TagTrackerPtr tag_tracker = nullptr);

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
    if (_tag_tracker == nullptr) {
      throw std::logic_error("Tag to Label is None");
    }
    return _tag_tracker->tagToLabel(tag);
  }

  const auto& processor() const { return _processor; }

  void setTagTracker(ner::utils::TagTrackerPtr tag_tracker) {
    _tag_tracker = std::move(tag_tracker);
  }

 private:
  void updateTokenTagCounter(
      const ArrayColumnBasePtr<std::string>& tokens,
      const ArrayColumnBasePtr<std::string>& tags) const {
    if (_tag_tracker->hasTokenTagCounter() && _target_column.has_value()) {
      for (size_t i = 0; i < tokens->numRows(); ++i) {
        std::vector<std::string> row_token_vectors = tokens->row(i).toVector();
        auto lower_cased_tokens =
            ner::utils::cleanAndLowerCase(row_token_vectors);

        for (size_t token_index = 0; token_index < row_token_vectors.size();
             ++token_index) {
          _tag_tracker->addTokenTag(lower_cased_tokens[token_index],
                                    tags->row(i)[token_index]);
        }
      }
    }
  }
  /*
   * _tokens_column : the column containing the string tokens
   * _target_column : the column containing the target tags
   */
  std::string _tokens_column;
  std::string _featurized_sentence_column;
  std::optional<std::string> _target_column;

  NerDyadicDataProcessor _processor;

  ner::utils::TagTrackerPtr _tag_tracker;
};

}  // namespace thirdai::data