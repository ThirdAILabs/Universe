#include "ExplanationMap.h"
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::data {

ExplanationMap::ExplanationMap(const ColumnMap& column_map) {
  if (column_map.numRows() != 1) {
    throw std::invalid_argument(
        "ExplanationMap can only be constructed from column maps with a single "
        "row.");
  }
  for (const auto& [name, column] : column_map) {
    std::string from_column = " from column '" + name + "'";
    if (auto token_array_col = ArrayColumnBase<uint32_t>::cast(column)) {
      for (uint32_t token : token_array_col->row(0)) {
        store(name, token, "token " + std::to_string(token) + from_column);
      }

    } else if (auto decimal_array_col = ArrayColumnBase<float>::cast(column)) {
      size_t feature_index = 0;
      for (float decimal : decimal_array_col->row(0)) {
        // stringstream formats floating points better by removing trailing
        // zeros.
        std::stringstream explanation;
        explanation << "decimal " << decimal << from_column;
        store(name, feature_index++, explanation.str());
      }

    } else if (auto str_col = ValueColumnBase<std::string>::cast(column)) {
      store(name, str_col->value(0), "column '" + name + "'");

    } else if (auto timestamp_col = ValueColumnBase<int64_t>::cast(column)) {
      int64_t timestamp = timestamp_col->value(0);
      store(name, /* feature_index= */ 0,
            "timestamp " + std::to_string(timestamp) + from_column);

    } else {
      throw std::invalid_argument("Unsupported input column type for RCA.");
    }
  }
}

const std::string& ExplanationMap::explain(const std::string& column,
                                           size_t feature_index) const {
  if (!_numerical_explanations.count(column)) {
    throw std::runtime_error("No explanations found for column '" + column +
                             "'.");
  }

  if (!_numerical_explanations.at(column).count(feature_index)) {
    throw std::runtime_error("No explanation found for feature " +
                             std::to_string(feature_index) + " in column '" +
                             column + "'.");
  }

  return _numerical_explanations.at(column).at(feature_index);
}

const std::string& ExplanationMap::explain(const std::string& column,
                                           const std::string& str) const {
  if (!_string_explanations.count(column)) {
    throw std::runtime_error("No explanations found for column '" + column +
                             "'.");
  }

  if (!_string_explanations.at(column).count(str)) {
    throw std::runtime_error("No explanation found for feature '" + str +
                             "' in column '" + column + "'.");
  }

  return _string_explanations.at(column).at(str);
}

void ExplanationMap::store(const std::string& column, size_t feature_index,
                           std::string explanation) {
  _numerical_explanations[column][feature_index] = std::move(explanation);
}

void ExplanationMap::store(const std::string& column, const std::string& str,
                           std::string explanation) {
  _string_explanations[column][str] = std::move(explanation);
}

std::vector<std::string> ExplanationMap::explanationsForColumn(
    const std::string& column) const {
  std::vector<std::string> explanations;
  if (_numerical_explanations.count(column)) {
    for (const auto& [_, explanation] : _numerical_explanations.at(column)) {
      explanations.push_back(explanation);
    }
  }

  if (_string_explanations.count(column)) {
    for (const auto& [_, explanation] : _string_explanations.at(column)) {
      explanations.push_back(explanation);
    }
  }

  return explanations;
}

}  // namespace thirdai::data