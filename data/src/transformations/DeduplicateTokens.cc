#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/DeduplicateTokens.h>
#include <exception>
#include <new>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
namespace thirdai::data {

ColumnMap DeduplicateTokens::apply(ColumnMap columns, State& state) const {
  (void)state;
  auto indices = columns.getArrayColumn<uint32_t>(_input_indices_column);
  ArrayColumnBasePtr<float> values = nullptr;
  if (_input_values_column) {
    values = columns.getArrayColumn<float>(*_input_values_column);
  }

  std::vector<std::vector<uint32_t>> deduped_indices(indices->numRows());
  std::vector<std::vector<float>> deduped_values(indices->numRows());

  std::exception_ptr error;

#pragma omp parallel for default(none)                       \
    shared(indices, values, deduped_indices, deduped_values, \
           error) if (columns.numRows() > 1)
  for (uint32_t i = 0; i < indices->numRows(); i++) {
    try {
      auto indices_row = indices->row(i);
      std::unordered_map<uint32_t, float> features;

      if (values) {
        auto values_row = values->row(i);
        for (uint32_t pos = 0; pos < indices_row.size(); pos++) {
          features[indices_row[pos]] += values_row[pos];
        }
      } else {
        for (const uint32_t token : indices_row) {
          features[token]++;
        }
      }

      deduped_indices[i].reserve(features.size());
      deduped_values[i].reserve(features.size());
      for (auto [index, value] : features) {
        deduped_indices[i].push_back(index);
        deduped_values[i].push_back(value);
      }
    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  columns.setColumn(
      _output_indices_column,
      ArrayColumn<uint32_t>::make(std::move(deduped_indices), indices->dim()));
  columns.setColumn(
      _output_values_column,
      ArrayColumn<float>::make(std::move(deduped_values), std::nullopt));

  return columns;
}

ar::ConstArchivePtr DeduplicateTokens::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("input_indices_column", ar::str(_input_indices_column));
  if (_input_values_column) {
    map->set("input_values_column", ar::str(*_input_values_column));
  }
  map->set("output_indices_column", ar::str(_output_indices_column));
  map->set("output_values_column", ar::str(_output_values_column));

  return map;
}

}  // namespace thirdai::data