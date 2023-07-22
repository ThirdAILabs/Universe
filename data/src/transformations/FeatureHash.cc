#include "FeatureHash.h"
#include <hashing/src/HashUtils.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::data {

FeatureHash::FeatureHash(std::vector<std::string> columns,
                         std::string output_indices, std::string output_values,
                         size_t dim)
    : _dim(dim),
      _columns(std::move(columns)),
      _output_indices(std::move(output_indices)),
      _output_values(std::move(output_values)) {}

ColumnMap FeatureHash::apply(ColumnMap columns, State& state) const {
  (void)state;

  std::vector<std::vector<uint32_t>> indices(columns.numRows());
  std::vector<std::vector<float>> values(columns.numRows());

  uint32_t column_salt = 0;

  for (const auto& name : _columns) {
    auto column = columns.getColumn(name);

    if (auto tokens =
            std::dynamic_pointer_cast<ValueColumn<uint32_t>>(column)) {
#pragma omp parallel for default(none) \
    shared(tokens, indices, values, column_salt)
      for (size_t i = 0; i < tokens->numRows(); i++) {
        indices[i].push_back(hash(tokens->value(i), column_salt));
        values[i].push_back(1.0);
      }
    } else if (auto token_arrays =
                   std::dynamic_pointer_cast<ArrayColumn<uint32_t>>(column)) {
#pragma omp parallel for default(none) \
    shared(token_arrays, indices, values, column_salt)
      for (size_t i = 0; i < token_arrays->numRows(); i++) {
        for (uint32_t token : token_arrays->row(i)) {
          indices[i].push_back(hash(token, column_salt));
          values[i].push_back(1.0);
        }
      }
    } else if (auto decimals =
                   std::dynamic_pointer_cast<ValueColumn<float>>(column)) {
#pragma omp parallel for default(none) \
    shared(decimals, indices, values, column_salt)
      for (size_t i = 0; i < decimals->numRows(); i++) {
        indices[i].push_back(hash(0, column_salt));
        values[i].push_back(decimals->value(i));
      }

    } else if (auto decimal_arrays =
                   std::dynamic_pointer_cast<ArrayColumn<float>>(column)) {
#pragma omp parallel for default(none) \
    shared(decimal_arrays, indices, values, column_salt)
      for (size_t i = 0; i < decimal_arrays->numRows(); i++) {
        size_t j = 0;
        for (float decimal : decimal_arrays->row(i)) {
          indices[i].push_back(hash(j++, column_salt));
          values[i].push_back(decimal);
        }
      }
    } else {
      throw std::invalid_argument(
          "Column '" + name +
          "' does not have a data type that can be feature hashed.");
    }

    column_salt++;
  }

  auto indices_col = ArrayColumn<uint32_t>::make(std::move(indices), _dim);
  auto values_col = ArrayColumn<float>::make(std::move(values));

  return ColumnMap(
      {{_output_indices, indices_col}, {_output_values, values_col}});
}

}  // namespace thirdai::data