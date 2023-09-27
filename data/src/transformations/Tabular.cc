#include "Tabular.h"
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/Transformation.h>
#include <proto/transformations.pb.h>
#include <string>
#include <vector>

namespace thirdai::data {

inline float parseFloat(const std::string& str) {
  if (str.empty()) {
    return 0.0;
  }
  return std::stof(str);
}

NumericalColumn::NumericalColumn(const proto::data::NumericalColumn& num_col)
    : name(num_col.name()),
      _min(num_col.min()),
      _max(num_col.max()),
      _binsize(num_col.binsize()),
      _num_bins(num_col.num_bins()),
      _salt(num_col.salt()) {}

inline uint32_t NumericalColumn::encode(const std::string& str_val) const {
  float val = parseFloat(str_val);

  uint32_t bin;
  if (val <= _min) {
    bin = 0;
  } else if (val >= _max) {
    bin = _num_bins - 1;
  } else {
    bin = (val - _min) / _binsize;
  }

  return hashing::combineHashes(bin, _salt);
}

proto::data::NumericalColumn* NumericalColumn::toProto() const {
  auto* num_col = new proto::data::NumericalColumn();

  num_col->set_name(name);
  num_col->set_min(_min);
  num_col->set_max(_max);
  num_col->set_binsize(_binsize);
  num_col->set_num_bins(_num_bins);
  num_col->set_salt(_salt);

  return num_col;
}

CategoricalColumn::CategoricalColumn(
    const proto::data::CategoricalColumn& cat_col)
    : name(cat_col.name()), _salt(cat_col.salt()) {}

inline uint32_t CategoricalColumn::encode(const std::string& val) const {
  return hashing::MurmurHash(val.data(), val.size(), _salt);
}

proto::data::CategoricalColumn* CategoricalColumn::toProto() const {
  auto* cat_col = new proto::data::CategoricalColumn();

  cat_col->set_name(name);
  cat_col->set_salt(_salt);

  return cat_col;
}

Tabular::Tabular(std::vector<NumericalColumn> numerical_columns,
                 std::vector<CategoricalColumn> categorical_columns,
                 std::string output_column, bool cross_column_pairgrams)
    : _numerical_columns(std::move(numerical_columns)),
      _categorical_columns(std::move(categorical_columns)),
      _output_column(std::move(output_column)),
      _cross_column_pairgrams(cross_column_pairgrams) {}

Tabular::Tabular(const proto::data::Tabular& tabular)
    : _output_column(tabular.output_column()),
      _cross_column_pairgrams(tabular.cross_column_pairgrams()) {
  for (const auto& cat_col : tabular.categorical_columns()) {
    _categorical_columns.emplace_back(cat_col);
  }

  for (const auto& num_col : tabular.numerical_columns()) {
    _numerical_columns.emplace_back(num_col);
  }
}

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
      row_tokens.push_back(num_col.encode(str_cols[col_idx++]->value(i)));
    }

    for (const auto& cat_col : _categorical_columns) {
      row_tokens.push_back(cat_col.encode(str_cols[col_idx++]->value(i)));
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

void Tabular::buildExplanationMap(const ColumnMap& input, State& state,
                                  ExplanationMap& explanations) const {
  (void)state;

  std::vector<std::pair<uint32_t, std::string>> tokens_and_explanations;

  for (const auto& num_col : _numerical_columns) {
    const std::string& item =
        input.getValueColumn<std::string>(num_col.name)->value(0);

    tokens_and_explanations.emplace_back(
        num_col.encode(item), "decimal " + item + " from " +
                                  explanations.explain(num_col.name, item));
  }

  for (const auto& cat_col : _categorical_columns) {
    const std::string& item =
        input.getValueColumn<std::string>(cat_col.name)->value(0);

    tokens_and_explanations.emplace_back(
        cat_col.encode(item), "category '" + item + "' from " +
                                  explanations.explain(cat_col.name, item));
  }

  if (_cross_column_pairgrams) {
    for (size_t i = 0; i < tokens_and_explanations.size(); i++) {
      for (size_t j = 0; j <= i; j++) {
        uint32_t pairgram = hashing::combineHashes(
            tokens_and_explanations[i].first, tokens_and_explanations[j].first);

        if (i == j) {
          explanations.store(_output_column, pairgram,
                             tokens_and_explanations[i].second);
        } else {
          explanations.store(_output_column, pairgram,
                             tokens_and_explanations[j].second + " and " +
                                 tokens_and_explanations[i].second);
        }
      }
    }
  } else {
    for (const auto& [token, exp] : tokens_and_explanations) {
      explanations.store(_output_column, token, exp);
    }
  }
}

proto::data::Transformation* Tabular::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* tabular = transformation->mutable_tabular();

  for (const auto& cat_col : _categorical_columns) {
    tabular->mutable_categorical_columns()->AddAllocated(cat_col.toProto());
  }

  for (const auto& num_col : _numerical_columns) {
    tabular->mutable_numerical_columns()->AddAllocated(num_col.toProto());
  }

  tabular->set_output_column(_output_column);
  tabular->set_cross_column_pairgrams(_cross_column_pairgrams);

  return transformation;
}

}  // namespace thirdai::data
