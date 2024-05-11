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
#include <memory>
#include <utility>

namespace thirdai::data {

NerTokenizerUnigram::NerTokenizerUnigram(
    std::string tokens_column, std::string featurized_sentence_column,
    std::optional<std::string> target_column, uint32_t fhr_dim,
    uint32_t dyadic_num_intervals,
    std::vector<dataset::TextTokenizerPtr> target_word_tokenizers)
    : _tokens_column(std::move(tokens_column)),
      _featurized_sentence_column(std::move(featurized_sentence_column)),
      _target_column(std::move(target_column)),
      _processor(std::move(target_word_tokenizers), dyadic_num_intervals) {
  _tokenizer_transformation = std::make_shared<TextTokenizer>(
      _featurized_sentence_column, _featurized_tokens_indices_column,
      _featurized_tokens_values_column,
      std::make_shared<dataset::NaiveSplitTokenizer>(
          dataset::NaiveSplitTokenizer()),
      std::make_shared<dataset::NGramEncoder>(dataset::NGramEncoder(1)), false,
      fhr_dim);
}

ColumnMap NerTokenizerUnigram::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto text_tokens = columns.getArrayColumn<std::string>(_tokens_column);

  ArrayColumnBasePtr<uint32_t> tags;
  if (_target_column) {
    tags = columns.getArrayColumn<uint32_t>(*_target_column);
  }

  auto sample_offsets = computeOffsets(text_tokens);

  std::vector<std::string> featurized_sentences(sample_offsets.back());
  std::vector<uint32_t> targets(sample_offsets.back());

  std::exception_ptr error;

  for (size_t i = 0; i < text_tokens->numRows(); i += 1) {
    RowView<std::string> row_tokens = text_tokens->row(i);
    size_t sample_offset = sample_offsets[i];

    for (size_t start = 0; start < row_tokens.size(); start += 1) {
      featurized_sentences[sample_offset] =
          _processor.processToken(row_tokens.toVector(), start);
      if (_target_column) {
        targets[sample_offset] = tags->row(i)[start];
      }

      sample_offset += 1;
    }
  }
  std::unordered_map<std::string, ColumnPtr> output_columns;

  output_columns[_featurized_sentence_column] =
      ValueColumn<std::string>::make(std::move(featurized_sentences));
  if (_target_column) {
    output_columns[*_target_column] =
        ValueColumn<uint32_t>::make(std::move(targets), tags->dim());
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