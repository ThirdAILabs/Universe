#include "StringHash.h"
#include <hashing/src/MurmurHash.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <dataset/src/utils/CsvParser.h>
#include <string>

namespace thirdai::data {

using dataset::parsers::CSV::parseLine;

ColumnMap StringHash::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto column = columns.getValueColumn<std::string>(_input_column_name);

  if (_delimiter) {
    std::vector<std::vector<uint32_t>> hashed_values(column->numRows());

#pragma omp parallel for default(none) shared(column, hashed_values)
    for (uint64_t i = 0; i < column->numRows(); i++) {
      auto values = parseLine(column->value(i), *_delimiter);
      hashed_values[i].reserve(values.size());

      for (const auto& value : values) {
        hashed_values[i].push_back(hash(value));
      }
    }

    auto output_column =
        ArrayColumn<uint32_t>::make(std::move(hashed_values), _output_range);

    columns.setColumn(_output_column_name, output_column);

  } else {
    std::vector<uint32_t> hashed_values(column->numRows());

#pragma omp parallel for default(none) \
    shared(column, hashed_values) if (column->numRows() > 1)
    for (uint64_t i = 0; i < column->numRows(); i++) {
      hashed_values[i] = hash(column->value(i));
    }

    auto output_column =
        ValueColumn<uint32_t>::make(std::move(hashed_values), _output_range);

    columns.setColumn(_output_column_name, output_column);
  }

  return columns;
}

uint32_t StringHash::hash(const std::string& str) const {
  uint32_t hash = hashing::MurmurHash(str.data(), str.length(), _seed);
  if (_output_range) {
    return hash % *_output_range;
  }
  return hash;
}

void StringHash::buildExplanationMap(const ColumnMap& input, State& state,
                                     ExplanationMap& explanations) const {
  (void)state;

  const auto& str =
      input.getValueColumn<std::string>(_input_column_name)->value(0);

  explanations.store(_output_column_name, hash(str),
                     "item '" + str + "' from " +
                         explanations.explain(_input_column_name, str));
}

}  // namespace thirdai::data