#include "Tabular.h"
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <string>
#include <vector>

namespace thirdai::data {

inline float parseFloat(const std::string& str) {
  if (str.empty()) {
    return 0.0;
  }
  return std::stof(str);
}

Tabular::Tabular(std::vector<NumericalColumn> numerical_columns,
                 std::vector<CategoricalColumn> categorical_columns,
                 std::string output_column, bool cross_column_pairgrams)
    : _numerical_columns(std::move(numerical_columns)),
      _categorical_columns(std::move(categorical_columns)),
      _output_column(std::move(output_column)),
      _cross_column_pairgrams(cross_column_pairgrams) {}

ColumnMap Tabular::apply(ColumnMap columns, State& state) const {
  (void)state;

  size_t num_rows = columns.numRows();

  std::vector<std::vector<uint32_t>> tokens(num_rows);

  std::vector<ValueColumnBasePtr<std::string>> str_cols;
  str_cols.reserve(_numerical_columns.size() + _categorical_columns.size());
  for (const auto& col : _numerical_columns) {
    str_cols.push_back(columns.getValueColumn<std::string>(col.name));
  }
  for (const auto& col : _categorical_columns) {
    str_cols.push_back(columns.getValueColumn<std::string>(col.name));
  }

#pragma omp parallel for default(none) \
    shared(tokens, str_cols) if (num_rows > 1)
  for (size_t i = 0; i < tokens.size(); i++) {
    size_t col_idx = 0;

    std::vector<uint32_t> row_tokens;
    tokens.reserve(str_cols.size());

    for (; col_idx < _numerical_columns.size(); col_idx++) {
      float val = parseFloat(str_cols[col_idx]->value(i));

      const auto& num_col = _numerical_columns[col_idx];
      uint32_t bin;
      if (val <= num_col.min) {
        bin = 0;
      } else if (val >= num_col.max) {
        bin = num_col.num_bins - 1;
      } else {
        bin = (val - num_col.min) / num_col.binsize;
      }

      row_tokens.push_back(hashing::combineHashes(bin, num_col.salt));
    }

    for (; col_idx < str_cols.size(); col_idx++) {
      const std::string& item = str_cols[col_idx]->value(i);
      tokens[i].push_back(hashing::MurmurHash(
          item.data(), item.size(), _categorical_columns[col_idx].salt));
    }

    if (_cross_column_pairgrams) {
      tokens[i] = dataset::token_encoding::pairgrams(row_tokens);
    } else {
      tokens[i] = std::move(row_tokens);
    }
  }

  columns.setColumn(_output_column,
                    ArrayColumn<uint32_t>::make(std::move(tokens)));

  return columns;
}

}  // namespace thirdai::data