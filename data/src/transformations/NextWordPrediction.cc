#include "NextWordPrediction.h"
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

NextWordPredictionConfig::NextWordPredictionConfig(std::string input_column, uint32_t vocab_size)
      :input_column(std::move(input_column)),
        vocab_size(vocab_size) {}


NextWordPrediction::NextWordPrediction(std::string input_column,
                                       std::string context_column,
                                       std::string target_column,
                                       std::optional<std::string> value_column)
    : _input_column(std::move(input_column)),
      _context_column(std::move(context_column)),
      _target_column(std::move(target_column)),
      _value_column(std::move(value_column)) {}

ColumnMap NextWordPrediction::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto texts = columns.getArrayColumn<uint32_t>(_input_column);
  auto sample_offsets = computeOffsets(texts);

  std::vector<std::vector<uint32_t>> contexts;
  contexts.assign(sample_offsets.back(), {});
  std::vector<uint32_t> targets(sample_offsets.back());
  // This is dummy for now, to fit in UDT input
  std::vector<std::vector<float>> values;
  if(_value_column){
    values.assign(sample_offsets.back(), {});
  }

  std::exception_ptr error;

#pragma omp parallel for default(none) \
    shared(texts, contexts, targets, values, sample_offsets, error)
  for (size_t i = 0; i < texts->numRows(); i += 1) {
    try {
      auto input_tokens = texts->row(i);

      size_t sample_offset = sample_offsets[i];
      for (size_t start = 1; start < input_tokens.size(); start += 1) {
        contexts[sample_offset] = input_tokens.range(0, start);
        if(_value_column){
          values[sample_offset] = std::vector<float>(start, 1.0);
        }
        targets[sample_offset] = input_tokens[start];

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
  // No need to pass in dimension, since we then make out output layers using this
  // atleast for mach, and then dimension would mismatch, since we ahve sparse input there.
  output_columns[_target_column] =
      ValueColumn<uint32_t>::make(std::move(targets), std::numeric_limits<uint32_t>::max());
  if(_value_column){
    output_columns[*_value_column] = 
      ArrayColumn<float>::make(std::move(values), std::nullopt);
  }

  return ColumnMap(output_columns);
}

std::vector<size_t> NextWordPrediction::computeOffsets(
    const ArrayColumnBasePtr<uint32_t>& texts) {
  std::vector<size_t> offsets(texts->numRows()+1);
  offsets[0] = 0;
  for (size_t i = 0; i < texts->numRows(); i++) {
    offsets[i+1] = offsets[i] + texts->row(i).size() - 1;
  }
  return offsets;
}

ar::ConstArchivePtr NextWordPrediction::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("input_column", ar::str(_input_column));
  map->set("context_column", ar::str(_context_column));
  map->set("target_column", ar::str(_target_column));

  return map;
}

NextWordPrediction::NextWordPrediction(const ar::Archive& archive)
    : _input_column(archive.str("input_column")),
      _context_column(archive.str("context_column")),
      _target_column(archive.str("target_column")) {}
}  // namespace thirdai::data