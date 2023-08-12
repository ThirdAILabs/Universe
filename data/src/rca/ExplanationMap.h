#pragma once

#include <data/src/ColumnMap.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace thirdai::data {

class ExplanationMap {
 public:
  explicit ExplanationMap(const ColumnMap& column_map);

  const std::string& explain(const std::string& column,
                             size_t feature_index) const;

  const std::string& explain(const std::string& column,
                             const std::string& str) const;

  void store(const std::string& column, size_t feature_index,
             std::string explanation);

  void store(const std::string& column, const std::string& str,
             std::string explanation);

  std::vector<std::string> explanationsForColumn(
      const std::string& column) const;

 private:
  using NumericalExplanations = std::unordered_map<size_t, std::string>;
  using StringExplanations = std::unordered_map<std::string, std::string>;

  std::unordered_map<std::string, NumericalExplanations>
      _numerical_explanations;

  std::unordered_map<std::string, StringExplanations> _string_explanations;
};

}  // namespace thirdai::data