#include "ConcatTokenArrayColumn.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::data {

ConcatTokenArrayColumn::ConcatTokenArrayColumn(std::string input_indices_column, std::string input_values_column,
                            std::string second_input_column, std::string output_indices_column, 
                            std::string output_value_column)
    : _input_indices_column(std::move(input_indices_column)),
      _input_values_column(std::move(input_values_column)),
      _second_input_column(std::move(second_input_column)),
      _output_indices_column(std::move(output_indices_column)),
      _output_value_column(std::move(output_value_column)) {}

ColumnMap ConcatTokenArrayColumn::apply(ColumnMap columns, State& state) const {
  (void)state;



  auto input_size = columns.getArrayColumn<uint32_t>(_input_indices_column)->numRows();
  auto second_input_size = columns.getArrayColumn<uint32_t>(_second_input_column)->numRows();
  if(input_size != second_input_size){
    throw std::invalid_argument("Input Columns doesnot have the same size.");
  }
  std::vector<std::vector<uint32_t>> indices(input_size, std::vector<uint32_t>{});
  std::vector<std::vector<float>> values(input_size, std::vector<float>{});

  auto input_indices = columns.getArrayColumn<uint32_t>(_input_indices_column);
  auto input_values = columns.getArrayColumn<float>(_input_values_column);
  auto second_input_indices = columns.getArrayColumn<uint32_t>(_second_input_column);

  #pragma omp parallel for default(none) \
      shared(input_size, indices, values, input_indices, input_values, second_input_indices)
  for (size_t i = 0; i < input_size; i += 1) {
    auto row_indices = input_indices->row(i);
    auto row_values = input_values->row(i);
    auto second_column_indices = second_input_indices->row(i);

    indices[i].insert(indices[i].end(),row_indices.begin(), row_indices.end());
    values[i].insert(values[i].end(),row_values.begin(), row_values.end());

    std::vector<float> ones(second_column_indices.size(), 1.0);

    indices[i].insert(indices[i].end(),second_column_indices.begin(), second_column_indices.end());
    values[i].insert(values[i].end(),ones.begin(), ones.end());
  }

  auto concat_indices = ArrayColumn<uint32_t>::make(std::move(indices), std::nullopt);
  auto concat_values = ArrayColumn<float>::make(std::move(values), std::nullopt);
  
  columns.setColumn(_output_indices_column, concat_indices);
  columns.setColumn(_output_value_column, concat_values);

  return columns;
}

ar::ConstArchivePtr ConcatTokenArrayColumn::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("input_indices_column", ar::str(_input_indices_column));
  map->set("input_values_column", ar::str(_input_values_column));
  map->set("second_input_column", ar::str(_second_input_column));
  map->set("output_indices_column", ar::str(_output_indices_column));
  map->set("output_value_column", ar::str(_output_value_column));

  return map;
}

ConcatTokenArrayColumn::ConcatTokenArrayColumn(const ar::Archive& archive)
    : _input_indices_column(archive.str("input_indices_column")),
      _input_values_column(archive.str("input_values_column")),
      _second_input_column(archive.str("second_input_column")),
      _output_indices_column(archive.str("output_indices_column")),
      _output_value_column(archive.str("output_value_column")) {}

}  // namespace thirdai::data