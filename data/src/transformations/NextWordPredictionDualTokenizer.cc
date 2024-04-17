#include "NextWordPredictionDualTokenizer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <utility>

namespace thirdai::data {

NextWordPredictionDualTokenizer::NextWordPredictionDualTokenizer(
    std::string input_column, std::string context_column,
    std::string target_column, std::string input_tokenizer,
    std::string output_tokenizer)
    : _input_column(std::move(input_column)),
      _context_column(std::move(context_column)),
      _target_column(std::move(target_column)),
      _input_tokenizer(automl::getTextTokenizerFromString(input_tokenizer)),
      _output_tokenizer(automl::getTextTokenizerFromString(output_tokenizer)) {}

ColumnMap NextWordPredictionDualTokenizer::apply(ColumnMap columns,
                                                 State& state) const {
  (void)state;

  // auto texts = columns.getArrayColumn<std::string>(_input_column);
  auto texts = columns.getValueColumn<std::string>(_input_column);
  // auto sample_offsets = computeOffsets(texts);

  std::vector<std::vector<uint32_t>> target_offsets(texts->numRows());

#pragma omp parallel for default(none) \
    shared(texts, _output_tokenizer, target_offsets)
  for (size_t i = 0; i < texts->numRows(); i += 1) {
    // std::string input_text = texts->row(i).data();
    auto input_text = texts->value(i);
    std::vector<uint32_t> offsets = _output_tokenizer->getOffsets(input_text);
    target_offsets[i] = offsets;
  }

  auto sample_offsets = computeOffsets(target_offsets);

  std::vector<std::vector<uint32_t>> contexts;
  contexts.assign(sample_offsets.back(), {});
  std::vector<uint32_t> targets(sample_offsets.back());

  std::exception_ptr error;

#pragma omp parallel for default(none)                                    \
    shared(texts, contexts, targets, _input_tokenizer, _output_tokenizer, \
               sample_offsets, target_offsets, error)
  for (size_t i = 0; i < texts->numRows(); i += 1) {
    try {
      std::string input_text = texts->value(i);
      std::vector<uint32_t> target_tokens =
          _output_tokenizer->tokenize(input_text);

      size_t sample_offset = sample_offsets[i];
      for (size_t start = 1; start < target_tokens.size(); start += 1) {
        uint32_t token_start = target_offsets[i][start];
        std::string context_window = input_text.substr(0, token_start);
        contexts[sample_offset] = _input_tokenizer->tokenize(context_window);
        targets[sample_offset] = target_tokens[start];
        sample_offset += 1;
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
  output_columns[_context_column] =
      ArrayColumn<uint32_t>::make(std::move(contexts), texts->dim());
  output_columns[_target_column] =
      ValueColumn<uint32_t>::make(std::move(targets), texts->dim());

  return ColumnMap(output_columns);
}

std::vector<size_t> NextWordPredictionDualTokenizer::computeOffsets(
    const std::vector<std::vector<uint32_t>>& target_offsets) const {
  std::vector<size_t> offsets(target_offsets.size() + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < target_offsets.size(); i++) {
    offsets[i + 1] = offsets[i] + target_offsets[i].size() - 1;
  }
  return offsets;
}

ar::ConstArchivePtr NextWordPredictionDualTokenizer::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("input_column", ar::str(_input_column));
  map->set("context_column", ar::str(_context_column));
  map->set("target_column", ar::str(_target_column));
  // auto input_tokenizer_archive =
  //     _input_tokenizer
  //         ->toArchive();  // Assuming toArchive exists in the tokenizer
  // auto output_tokenizer_archive = _output_tokenizer->toArchive();
  // map->set("input_tokenizer", input_tokenizer_archive);
  // map->set("output_tokenizer", output_tokenizer_archive);

  return map;
}

NextWordPredictionDualTokenizer::NextWordPredictionDualTokenizer(
    const ar::Archive& archive)
    : _input_column(archive.str("input_column")),
      _context_column(archive.str("context_column")),
      _target_column(archive.str("target_column")) {}
}  // namespace thirdai::data