#include "FeatureHash.h"
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::data {

FeatureHash::FeatureHash(std::vector<std::string> input_columns,
                         std::string output_indices_column,
                         std::string output_values_column, size_t hash_range)
    : _hash_range(hash_range),
      _input_columns(std::move(input_columns)),
      _output_indices_column(std::move(output_indices_column)),
      _output_values_column(std::move(output_values_column)) {}

ColumnMap FeatureHash::apply(ColumnMap columns, State& state) const {
  (void)state;

  std::vector<std::vector<uint32_t>> indices(columns.numRows());
  std::vector<std::vector<float>> values(columns.numRows());

  for (const auto& name : _input_columns) {
    auto column = columns.getColumn(name);

    uint32_t column_salt = columnSalt(name);

    if (auto token_arrays =
            std::dynamic_pointer_cast<ArrayColumnBase<uint32_t>>(column)) {
#pragma omp parallel for default(none) \
    shared(token_arrays, indices, values, column_salt)
      for (size_t i = 0; i < token_arrays->numRows(); i++) {
        for (uint32_t token : token_arrays->row(i)) {
          indices[i].push_back(hash(token, column_salt));
          values[i].push_back(1.0);
        }
      }
    } else if (auto decimal_arrays =
                   std::dynamic_pointer_cast<ArrayColumnBase<float>>(column)) {
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

  auto indices_col =
      ArrayColumn<uint32_t>::make(std::move(indices), _hash_range);
  auto values_col = ArrayColumn<float>::make(std::move(values), std::nullopt);

  return ColumnMap({{_output_indices_column, indices_col},
                    {_output_values_column, values_col}});
}

void FeatureHash::explainFeatures(const ColumnMap& input, State& state,
                                  FeatureExplainations& explainations) const {
  (void)state;

  ColumnFeatureExplainations feature_explainations;

  for (const auto& name : _input_columns) {
    auto column = input.getColumn(name);

    uint32_t column_salt = columnSalt(name);

    if (auto token_arrays =
            std::dynamic_pointer_cast<ArrayColumnBase<uint32_t>>(column)) {
      for (size_t i = 0; i < token_arrays->numRows(); i++) {
        for (uint32_t token : token_arrays->row(i)) {
          uint32_t feature = hash(token, column_salt);

          if (feature_explainations.count(feature)) {
            feature_explainations[feature] +=
                " " + explainations.explainFeature(name, token);
          } else {
            feature_explainations[feature] =
                explainations.explainFeature(name, token);
          }
        }
      }
    } else if (auto decimal_arrays =
                   std::dynamic_pointer_cast<ArrayColumnBase<float>>(column)) {
      for (size_t i = 0; i < decimal_arrays->numRows(); i++) {
        size_t row_len = decimal_arrays->row(i).size();
        for (size_t feature_idx = 0; feature_idx < row_len; feature_idx++) {
          uint32_t feature = hash(feature_idx, column_salt);

          if (feature_explainations.count(feature)) {
            feature_explainations[feature] +=
                " " + explainations.explainFeature(name, feature_idx);
          } else {
            feature_explainations[feature] =
                explainations.explainFeature(name, feature_idx);
          }
        }
      }
    } else {
      throw std::invalid_argument(
          "Column '" + name +
          "' does not have a data type that can be feature hashed.");
    }
  }

  for (const auto& [feature, explaination] : feature_explainations) {
    explainations.addFeatureExplaination(_output_indices_column, feature,
                                         explaination);
  }
}

}  // namespace thirdai::data