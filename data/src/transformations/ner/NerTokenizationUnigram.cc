#include "NerTokenizationUnigram.h"
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/ner/NerDyadicDataProcessor.h>
#include <data/src/transformations/ner/NerTokenFromStringArray.h>
#include <data/src/transformations/ner/NerTokenTagCounter.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::data {

std::string trimPunctuation(const std::string& str) {
  const std::string punctuation = ".,?-!;:[]{}&%";
  size_t start = str.find_first_not_of(punctuation);
  if (start == std::string::npos) {
    return str;
  }
  size_t end = str.find_last_not_of(punctuation);
  return str.substr(start, end - start + 1);
}

std::vector<std::string> cleanAndLowerCase(
    const std::vector<std::string>& tokens) {
  /*
   * Converts the tokens to lower case and trims punctuations.
   */
  auto lower_tokens = tokens;
  for (auto& token : lower_tokens) {
    for (char& c : token) {
      c = std::tolower(c);
    }
  }
  for (auto& token : lower_tokens) {
    token = trimPunctuation(token);
  }
  return lower_tokens;
}

NerTokenizerUnigram::NerTokenizerUnigram(
    std::string tokens_column, std::string featurized_sentence_column,
    std::optional<std::string> target_column,
    std::optional<uint32_t> target_dim, uint32_t dyadic_num_intervals,
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
    std::optional<FeatureEnhancementConfig> feature_enhancement_config,
    std::unordered_map<std::string, uint32_t> tag_to_label,
    std::unordered_set<std::string> ignored_tags,
    ner::TokenTagCounterPtr token_tag_counter)
    : _tokens_column(std::move(tokens_column)),
      _featurized_sentence_column(std::move(featurized_sentence_column)),
      _target_column(std::move(target_column)),
      _target_dim(target_dim),
      _processor(std::move(target_word_tokenizers), dyadic_num_intervals,
                 std::move(feature_enhancement_config),
                 target_column == std::nullopt),
      _tag_to_label(std::move(tag_to_label)),
      _ignored_tags(std::move(ignored_tags)),
      _token_tag_counter(std::move(token_tag_counter)) {}

ColumnMap NerTokenizerUnigram::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto text_tokens = columns.getArrayColumn<std::string>(_tokens_column);

  ArrayColumnBasePtr<std::string> tags;
  if (_target_column) {
    tags = columns.getArrayColumn<std::string>(*_target_column);
  }

  auto sample_offsets = computeOffsets(text_tokens);

  std::vector<std::string> featurized_sentences(sample_offsets.back());
  std::vector<uint32_t> targets(sample_offsets.back());

  std::exception_ptr error;

  if (_token_tag_counter != nullptr && _target_column.has_value()) {
    for (size_t i = 0; i < text_tokens->numRows(); i += 1) {
      std::vector<std::string> row_token_vectors =
          text_tokens->row(i).toVector();

      auto lower_cased_tokens = cleanAndLowerCase(row_token_vectors);
      for (size_t token_index = 0; token_index < row_token_vectors.size();
           token_index++) {
        if (!_ignored_tags.count(tags->row(i)[token_index])) {
          _token_tag_counter->addTokenTag(lower_cased_tokens[token_index],
                                          tags->row(i)[token_index]);
        }
      }
    }
  }

#pragma omp parallel for default(none)                                       \
    shared(text_tokens, sample_offsets, featurized_sentences, targets, tags, \
           error) if (text_tokens->numRows() > 1)
  for (size_t i = 0; i < text_tokens->numRows(); i += 1) {
    try {
      size_t sample_offset = sample_offsets[i];
      std::vector<std::string> row_token_vectors =
          text_tokens->row(i).toVector();

      auto lower_cased_tokens = cleanAndLowerCase(row_token_vectors);

      for (size_t target = 0; target < row_token_vectors.size(); target++) {
        size_t featurized_sentence_offset = sample_offset + target;
        featurized_sentences[featurized_sentence_offset] =
            _processor.processToken(row_token_vectors, target,
                                    lower_cased_tokens);

        if (_token_tag_counter != nullptr) {
          featurized_sentences[featurized_sentence_offset] +=
              _token_tag_counter->getTokenEncoding(lower_cased_tokens[target]);
        }

        if (_target_column) {
          if (row_token_vectors.size() != tags->row(i).size()) {
            std::stringstream error_message;
            error_message
                << "Mismatch between the number of tokens and tags in row " << i
                << ":\n"
                << "  - Number of tokens: " << row_token_vectors.size() << "\n"
                << "  - Number of tags: " << tags->row(i).size() << "\n"
                << "Please ensure each token has a corresponding tag.";
            throw std::out_of_range(error_message.str());
          }
          if (!_tag_to_label.empty()) {
            targets[featurized_sentence_offset] =
                findTagValueForString(tags->row(i)[target]);
          } else {
            targets[featurized_sentence_offset] =
                std::stoi(tags->row(i)[target]);
          }
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
  if (_target_column && _target_dim) {
    output_columns[*_target_column] =
        ValueColumn<uint32_t>::make(std::move(targets), _target_dim.value());
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
  if (_target_dim) {
    map->set("target_dim", ar::u64(*_target_dim));
  }

  map->set("processor", _processor.toArchive());

  map->set("tag_to_label",
           ar::mapStrU64({_tag_to_label.begin(), _tag_to_label.end()}));

  map->set("ignored_tags",
           ar::vecStr({_ignored_tags.begin(), _ignored_tags.end()}));

  return map;
}

NerTokenizerUnigram::NerTokenizerUnigram(const ar::Archive& archive)
    : _tokens_column(archive.str("tokens_column")),
      _featurized_sentence_column(archive.str("featurized_sentence_column")),
      _target_column(archive.getOpt<ar::Str>("target_column")),
      _target_dim(archive.getOpt<ar::U64>("target_dim")),
      _processor(NerDyadicDataProcessor(*archive.get("processor"))) {
  if (archive.contains("tag_to_label")) {
    const auto& tag_to_label = archive.getAs<ar::MapStrU64>("tag_to_label");
    _tag_to_label = {tag_to_label.begin(), tag_to_label.end()};
  }
  if (archive.contains("ignored_tags")) {
    const auto& ignored_tags = archive.getAs<ar::VecStr>("ignored_tags");
    _ignored_tags = {ignored_tags.begin(), ignored_tags.end()};
  }
}

}  // namespace thirdai::data