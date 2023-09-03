#include "Tabular.h"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/Transformation.h>
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
    row_tokens.reserve(str_cols.size());

    for (const auto& num_col : _numerical_columns) {
      float val = parseFloat(str_cols[col_idx++]->value(i));

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

    for (const auto& cat_col : _categorical_columns) {
      const std::string& item = str_cols[col_idx++]->value(i);

      row_tokens.push_back(
          hashing::MurmurHash(item.data(), item.size(), cat_col.salt));
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

template <class Archive>
void Tabular::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _numerical_columns,
          _categorical_columns, _output_column, _cross_column_pairgrams);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::Tabular)