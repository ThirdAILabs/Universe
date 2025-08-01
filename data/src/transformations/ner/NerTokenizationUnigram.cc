#include "NerTokenizationUnigram.h"
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/ner/NerDyadicDataProcessor.h>
#include <data/src/transformations/ner/NerTokenFromStringArray.h>
#include <data/src/transformations/ner/utils/TagTracker.h>
#include <data/src/transformations/ner/utils/utils.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::data {

void throwTokenTagSizeMismatchError(uint32_t row_id, uint32_t tokens_size,
                                    uint32_t tags_size) {
  std::stringstream error_message;
  error_message << "Mismatch between the number of tokens and tags in row "
                << row_id << ":\n"
                << "  - Number of tokens: " << tokens_size << "\n"
                << "  - Number of tags: " << tags_size << "\n"
                << "Please ensure each token has a corresponding tag.";
  throw std::out_of_range(error_message.str());
}

NerTokenizerUnigram::NerTokenizerUnigram(
    std::string tokens_column, std::string featurized_sentence_column,
    std::optional<std::string> target_column, uint32_t dyadic_num_intervals,
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
    std::optional<FeatureEnhancementConfig> feature_enhancement_config,
    ner::utils::TagTrackerPtr tag_tracker)
    : _tokens_column(std::move(tokens_column)),
      _featurized_sentence_column(std::move(featurized_sentence_column)),
      _target_column(std::move(target_column)),
      _processor(std::move(target_word_tokenizers), dyadic_num_intervals,
                 std::move(feature_enhancement_config),
                 !_target_column.has_value()),
      _tag_tracker(std::move(tag_tracker)) {}

ColumnMap NerTokenizerUnigram::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto text_tokens = columns.getArrayColumn<std::string>(_tokens_column);
  auto sample_offsets = computeOffsets(text_tokens);

  ArrayColumnBasePtr<std::string> tags;
  if (_target_column) {
    tags = columns.getArrayColumn<std::string>(*_target_column);
  }

  // a no-op if token_tag_counter is null or if featurizing for inference
  updateTokenTagCounter(text_tokens, tags);

  std::vector<std::string> featurized_sentences(sample_offsets.back());
  std::vector<uint32_t> targets(sample_offsets.back());

  std::exception_ptr error;

#pragma omp parallel for default(none)                                       \
    shared(text_tokens, sample_offsets, featurized_sentences, targets, tags, \
           error) if (text_tokens->numRows() > 1)
  for (size_t i = 0; i < text_tokens->numRows(); i += 1) {
    try {
      size_t sample_offset = sample_offsets[i];
      std::vector<std::string> row_token_vectors =
          text_tokens->row(i).toVector();
      auto lower_cased_tokens =
          ner::utils::cleanAndLowerCase(row_token_vectors);

      // processing the tokens in the sentence
      for (size_t target = 0; target < row_token_vectors.size(); target++) {
        size_t featurized_sentence_offset = sample_offset + target;
        featurized_sentences[featurized_sentence_offset] =
            _processor.processToken(row_token_vectors, target,
                                    lower_cased_tokens);
        if (_tag_tracker->hasTokenTagCounter()) {
          featurized_sentences[featurized_sentence_offset] +=
              _tag_tracker->getTokenEncoding(lower_cased_tokens[target]);
        }
        if (_target_column) {
          if (row_token_vectors.size() != tags->row(i).size()) {
            throwTokenTagSizeMismatchError(i, row_token_vectors.size(),
                                           tags->row(i).size());
          }
          targets[featurized_sentence_offset] =
              findTagValueForString(tags->row(i)[target]);
        }
      }
    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }
  if (error) {
    std::rethrow_exception(error);
  }
  std::unordered_map<std::string, ColumnPtr> output_columns;

  output_columns[_featurized_sentence_column] =
      ValueColumn<std::string>::make(std::move(featurized_sentences));
  if (_target_column) {
    output_columns[*_target_column] = ValueColumn<uint32_t>::make(
        std::move(targets), _tag_tracker->numLabels());
  }

  return ColumnMap(output_columns);
}

ar::ConstArchivePtr NerTokenizerUnigram::toArchive() const {
  auto map = ar::Map::make();
  map->set("type", ar::str(type()));

  map->set("tokens_column", ar::str(_tokens_column));
  map->set("featurized_sentence_column", ar::str(_featurized_sentence_column));

  if (_target_column) {
    map->set("target_column", ar::str(*_target_column));
  }

  map->set("processor", _processor.toArchive());
  return map;
}

NerTokenizerUnigram::NerTokenizerUnigram(const ar::Archive& archive)
    : _tokens_column(archive.str("tokens_column")),
      _featurized_sentence_column(archive.str("featurized_sentence_column")),
      _target_column(archive.getOpt<ar::Str>("target_column")),
      _processor(NerDyadicDataProcessor(*archive.get("processor"))) {}

}  // namespace thirdai::data