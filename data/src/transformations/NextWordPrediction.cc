#include "NextWordPrediction.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/CsvParser.h>
#include <utility>

namespace thirdai::data {

using dataset::parsers::CSV::parseLine;

NextWordPrediction::NextWordPrediction(std::string input_column,
                                       std::string context_column,
                                       std::string target_column,
                                       std::optional<std::string> text_input_column)
    : _input_column(std::move(input_column)),
      _context_column(std::move(context_column)),
      _target_column(std::move(target_column)),
      _text_input_column(std::move(text_input_column)) {}

ColumnMap NextWordPrediction::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto input_tokens = columns.getArrayColumn<uint32_t>(_input_column);
  ValueColumnBasePtr<std::string> input_texts = nullptr;
  if(_text_input_column){
    input_texts = columns.getValueColumn<std::string>(*_text_input_column);
  }

  auto sample_offsets = computeOffsets(input_tokens);

  std::vector<std::vector<uint32_t>> token_contexts;
  std::vector<std::vector<std::string>> text_contexts;

  if(_text_input_column){
    text_contexts.assign(sample_offsets.back(), {});
  }else{
    token_contexts.assign(sample_offsets.back(), {});
  }


  std::vector<uint32_t> targets(sample_offsets.back());

  std::exception_ptr error;

#pragma omp parallel for default(none) \
    shared(input_tokens, token_contexts, text_contexts, targets, input_texts, sample_offsets, error)
  for (size_t i = 0; i < input_tokens->numRows(); i += 1) {
    try {
      auto input_token_row = input_tokens->row(i);

      std::vector<std::string> text_row;
      if(_text_input_column){
        auto input_text_row = parseLine(input_texts->value(i), ' ');
        text_row.insert(text_row.end(), input_text_row.begin(), input_text_row.end());
      }

      std::vector<uint32_t> tokens_row;
      tokens_row.insert(tokens_row.end(), input_token_row.begin(), input_token_row.end());

      size_t sample_offset = sample_offsets[i];
      for (size_t start = 1; start < tokens_row.size(); start += 1) {
        if(_text_input_column){
          text_contexts[sample_offset] = {text_row.begin(), text_row.begin() + start};
        }else{
          token_contexts[sample_offset] = {tokens_row.begin(), tokens_row.begin() + start};
        }
        targets[sample_offset] = tokens_row[start];
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
  if(_text_input_column){
    output_columns[_context_column] =
        ArrayColumn<std::string>::make(std::move(text_contexts));
  }else{
    output_columns[_context_column] =
        ArrayColumn<uint32_t>::make(std::move(token_contexts), input_tokens->dim());
  }
  output_columns[_target_column] =
      ValueColumn<uint32_t>::make(std::move(targets), input_tokens->dim());

  return ColumnMap(output_columns);
}

std::vector<size_t> NextWordPrediction::computeOffsets(
    const ArrayColumnBasePtr<uint32_t>& texts) const {
  std::vector<size_t> offsets(texts->numRows() + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < texts->numRows(); i++) {
    offsets[i + 1] = offsets[i] + texts->row(i).size() - 1;
  }
  return offsets;
}

ar::ConstArchivePtr NextWordPrediction::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("input_column", ar::str(_input_column));
  map->set("context_column", ar::str(_context_column));
  map->set("target_column", ar::str(_target_column));
  if(_text_input_column){
    map->set("text_input_column", ar::str(*_text_input_column));
  }

  return map;
}

NextWordPrediction::NextWordPrediction(const ar::Archive& archive)
    : _input_column(archive.str("input_column")),
      _context_column(archive.str("context_column")),
      _target_column(archive.str("target_column")),
      _text_input_column(archive.getOpt<ar::Str>("text_input_column")) {}
}  // namespace thirdai::data