#pragma once

#include <data/src/ColumnMap.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace thirdai::data {

/**
 * This class stores explanations as for the data as transformations are applied
 * to it. It is initialized from a ColumnMap with a single row. At the begining
 * it has simple explanantions for the raw data types that are passed in.
 * Strings are just explained by the column name they occur in, tokens and
 * decimals say "token 8 from column 'col'", etc. As transformations
 * 'buildExplanationMap' methods are invoked on the data they should add
 * explanations for the features they create. These explanations should build
 * upon the explanations for the features that the transformation uses. The goal
 * is to create a list of explanations for the output features that are in terms
 * of the input data, so that it is clear how the input data contributed to the
 * output.
 *
 * For example if a CrossColumnPairgram transformation is invoked on data with
 * the following explanations:
 *
 * Column 'a':
 *    8  -> "token 8 from column 'a'"
 *
 * Column 'b':
 *    18 -> "token 18 from column 'b'"
 *
 * Then it should create a new entry in the explanation map:
 *
 * Column 'output':
 *    hash(8, 18) -> "token 8 from column 'a' and token 18 from column 'b'"
 */
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