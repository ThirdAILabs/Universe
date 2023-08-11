#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace thirdai::data {

using ColumnFeatureExplainations = std::unordered_map<size_t, std::string>;

class FeatureExplainations {
 public:
  std::string explainNumeric(const std::string& column, size_t feature_index) {
    if (!_column_feature_explainations.count(column)) {
      throw std::runtime_error("No explainations found for column '" + column +
                               "'.");
    }

    if (!_column_feature_explainations.at(column).count(feature_index)) {
      throw std::runtime_error("No explaination found for feature " +
                               std::to_string(feature_index) + " in column '" +
                               column + "'.");
    }

    return _column_feature_explainations.at(column).at(feature_index);
  }

  std::string explainString(const std::string& column) {
    return explainNumeric(column, 0);
  }

  void addFeatureExplaination(const std::string& column, size_t feature_index,
                              const std::string& explaination) {
    _column_feature_explainations[column][feature_index] = explaination;
  }

 private:
  std::unordered_map<std::string, ColumnFeatureExplainations>
      _column_feature_explainations;
};

}  // namespace thirdai::data