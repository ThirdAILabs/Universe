#include "FeatureHash.h"
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

    if (auto token_arrays = ArrayColumnBase<uint32_t>::cast(column)) {
#pragma omp parallel for default(none) shared( \
        token_arrays, indices, values, column_salt) if (columns.numRows() > 1)
      for (size_t i = 0; i < token_arrays->numRows(); i++) {
        for (uint32_t token : token_arrays->row(i)) {
          indices[i].push_back(hash(token, column_salt));
          values[i].push_back(1.0);
        }
      }
    } else if (auto decimal_arrays = ArrayColumnBase<float>::cast(column)) {
#pragma omp parallel for default(none)      \
    shared(decimal_arrays, indices, values, \
               column_salt) if (columns.numRows() > 1)
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

  columns.setColumn(_output_indices_column, indices_col);
  columns.setColumn(_output_values_column, values_col);

  return columns;
}

void FeatureHash::buildExplanationMap(const ColumnMap& input, State& state,
                                      ExplanationMap& explanations) const {
  (void)state;

  std::unordered_map<size_t, std::string> feature_explanations;

  for (const auto& name : _input_columns) {
    auto column = input.getColumn(name);

    uint32_t column_salt = columnSalt(name);

    if (auto token_arrays = ArrayColumnBase<uint32_t>::cast(column)) {
      for (size_t i = 0; i < token_arrays->numRows(); i++) {
        for (uint32_t token : token_arrays->row(i)) {
          uint32_t feature = hash(token, column_salt);

          // Concatenate the explanations for the features if there are hash
          // colisions, since the output feature comes from both of the input
          // features.
          if (feature_explanations.count(feature)) {
            feature_explanations[feature] +=
                " " + explanations.explain(name, token);
          } else {
            feature_explanations[feature] = explanations.explain(name, token);
          }
        }
      }
    } else if (auto decimal_arrays = ArrayColumnBase<float>::cast(column)) {
      for (size_t i = 0; i < decimal_arrays->numRows(); i++) {
        size_t row_len = decimal_arrays->row(i).size();
        for (size_t feature_idx = 0; feature_idx < row_len; feature_idx++) {
          uint32_t feature = hash(feature_idx, column_salt);

          // Concatenate the explanations for the features if there are hash
          // colisions, since the output feature comes from both of the input
          // features.
          if (feature_explanations.count(feature)) {
            feature_explanations[feature] +=
                " " + explanations.explain(name, feature_idx);
          } else {
            feature_explanations[feature] =
                explanations.explain(name, feature_idx);
          }
        }
      }
    } else {
      throw std::invalid_argument(
          "Column '" + name +
          "' does not have a data type that can be feature hashed.");
    }
  }

  for (const auto& [feature, explanation] : feature_explanations) {
    explanations.store(_output_indices_column, feature, explanation);
  }
}

ar::ConstArchivePtr FeatureHash::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("input_columns", ar::vecStr(_input_columns));
  map->set("output_indices_column", ar::str(_output_indices_column));
  map->set("output_values_column", ar::str(_output_values_column));
  map->set("hash_range", ar::u64(_hash_range));

  return map;
}

FeatureHash::FeatureHash(const ar::Archive& archive)
    : _hash_range(archive.u64("hash_range")),
      _input_columns(archive.getAs<ar::VecStr>("input_columns")),
      _output_indices_column(archive.str("output_indices_column")),
      _output_values_column(archive.str("output_values_column")) {}

template void FeatureHash::serialize(cereal::BinaryInputArchive&);
template void FeatureHash::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void FeatureHash::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _hash_range, _input_columns,
          _output_indices_column, _output_values_column);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::FeatureHash)