#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/DeduplicateTokens.h>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
namespace thirdai::data {

static std::pair<std::vector<uint32_t>, std::vector<float>> deduplicate(
    RowView<uint32_t> indices, std::optional<RowView<float>> values) {
  std::unordered_map<uint32_t, float> features;
  for (uint32_t i = 0; i < indices.size(); i++) {
    features[indices[i]] += values ? (*values)[i] : 1;
  }
  std::vector<uint32_t> deduped_indices;
  deduped_indices.reserve(features.size());
  std::vector<float> deduped_values;
  deduped_values.reserve(features.size());
  for (auto [index, value] : features) {
    deduped_indices.push_back(index);
    deduped_values.push_back(value);
  }
  return std::make_pair(std::move(deduped_indices), std::move(deduped_values));
}

ColumnMap DeduplicateTokens::apply(ColumnMap columns, State& state) const {
  (void)state;
  auto indices = columns.getArrayColumn<uint32_t>(_input_indices_column);
  std::optional<ArrayColumnBasePtr<float>> values;
  if (_input_values_column) {
    values = columns.getArrayColumn<float>(*_input_values_column);
  }

  std::vector<std::vector<uint32_t>> deduped_indices(indices->numRows());
  std::vector<std::vector<float>> deduped_values(indices->numRows());

#pragma omp parallel for default(none) \
    shared(indices, values, deduped_indices, deduped_values)
  for (uint32_t i = 0; i < indices->numRows(); i++) {
    std::optional<RowView<float>> row_values;
    if (values) {
      row_values = (*values)->row(i);
    }

    auto [deduped_row_indices, deduped_row_values] =
        deduplicate(indices->row(i), row_values);
    deduped_indices[i] = std::move(deduped_row_indices);
    deduped_values[i] = std::move(deduped_row_values);
  }

  columns.setColumn(
      _output_indices_column,
      ArrayColumn<uint32_t>::make(std::move(deduped_indices), indices->dim()));
  columns.setColumn(
      _output_values_column,
      ArrayColumn<float>::make(std::move(deduped_values), std::nullopt));

  return columns;
}

}  // namespace thirdai::data