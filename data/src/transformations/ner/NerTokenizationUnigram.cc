#include "NerTokenizationUnigram.h"
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/NerTokenFromStringArray.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/ner/UnigramDataProcessor.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <utility>

namespace thirdai::data {

NerTokenizerUnigram::NerTokenizerUnigram(
    std::string tokens_column, std::string featurized_sentence_column,
    std::optional<std::string> target_column,
    std::optional<uint32_t> target_dim, uint32_t fhr_dim,
    uint32_t dyadic_num_intervals,
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers,
    std::optional<std::unordered_map<std::string, uint32_t>> tag_to_label)
    : _tokens_column(std::move(tokens_column)),
      _featurized_sentence_column(std::move(featurized_sentence_column)),
      _target_column(std::move(target_column)),
      _target_dim(target_dim),
      _processor(std::move(target_word_tokenizers), dyadic_num_intervals),
      _tag_to_label(std::move(tag_to_label)) {
  /*
   * Target Word Tokenizers are used to tokenize the target token. They are used
   * for generating extra features for the target token.
   */
  _tokenizer_transformation = std::make_shared<TextTokenizer>(
      /*input_column=*/_featurized_sentence_column,
      /*output_indices=*/_featurized_tokens_indices_column,
      /*output_values=*/std::nullopt,
      /*tokenizer=*/
      std::make_shared<dataset::NaiveSplitTokenizer>(
          dataset::NaiveSplitTokenizer()),
      /*encoder=*/
      std::make_shared<dataset::NGramEncoder>(dataset::NGramEncoder(1)), false,
      fhr_dim);
}

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

  /*
   * This a nested pragma loop. The outer loop parallelizes over the different
   * sentences and the inner loop performs parallelization over the tokens in a
   * sentence. The inner loop only activates when there is a single sentence in
   * the column map.
   * TODO(Shubh) : Convert to a single for loop by using offsets
   * for the tokens inside samples.
   */

#pragma omp parallel for default(none)                                       \
    shared(text_tokens, sample_offsets, featurized_sentences, targets, tags, \
           error) if (text_tokens->numRows() > 1)
  for (size_t i = 0; i < text_tokens->numRows(); i += 1) {
    try {
      size_t sample_offset = sample_offsets[i];
      std::vector<std::string> row_token_vectors =
          text_tokens->row(i).toVector();
#pragma omp parallel for default(none)                                         \
    shared(text_tokens, sample_offset, featurized_sentences, targets, tags, i, \
           row_token_vectors) if (text_tokens->numRows() <= 1)
      for (size_t start = 0; start < row_token_vectors.size(); start += 1) {
        size_t featurized_sentence_offset = sample_offset + start;
        featurized_sentences[featurized_sentence_offset] =
            _processor.processToken(row_token_vectors, start);
        if (_target_column) {
          if (_tag_to_label.has_value()) {
            targets[featurized_sentence_offset] =
                findTagValueForString(tags->row(i)[start]);
          } else {
            targets[featurized_sentence_offset] =
                std::stoi(tags->row(i)[start]);
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
  if (_target_column) {
    output_columns[*_target_column] =
        ValueColumn<uint32_t>::make(std::move(targets), _target_dim.value());
  }

  ColumnMap processed_column_map = ColumnMap(output_columns);

  // this applies inplace transformation to the column map and tokenizes the
  // sentences into indices and values array pairs.
  return _tokenizer_transformation->apply(processed_column_map, state);
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
  map->set("tokenizer", _tokenizer_transformation->toArchive());
  return map;
}

NerTokenizerUnigram::NerTokenizerUnigram(const ar::Archive& archive)
    : _tokens_column(archive.str("tokens_column")),
      _featurized_sentence_column(archive.str("featurized_sentence_column")),
      _target_column(archive.getOpt<ar::Str>("target_column")),
      _processor(SimpleDataProcessor(*archive.get("processor"))) {
  _tokenizer_transformation =
      TextTokenizer::fromArchive(*archive.get("tokenizer"));
}
}  // namespace thirdai::data