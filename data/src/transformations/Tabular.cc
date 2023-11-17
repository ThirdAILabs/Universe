#include "Tabular.h"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/HashUtils.h>
#include <hashing/src/MurmurHash.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
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

inline uint32_t CategoricalColumn::encode(const std::string& val) const {
  return hashing::MurmurHash(val.data(), val.size(), _salt);
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

ar::ConstArchivePtr Tabular::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  auto numerical_columns = ar::List::make();
  for (const auto& num_col : _numerical_columns) {
    numerical_columns->append(num_col.toArchive());
  }
  map->set("numerical_columns", numerical_columns);

  auto categorical_columns = ar::List::make();
  for (const auto& cat_col : _numerical_columns) {
    categorical_columns->append(cat_col.toArchive());
  }
  map->set("categorical_columns", categorical_columns);

  map->set("output_column", ar::str(_output_column));
  map->set("cross_column_pairgrams", ar::boolean(_cross_column_pairgrams));

  return map;
}

Tabular::Tabular(const ar::Archive& archive)
    : _output_column(archive.str("output_column")),
      _cross_column_pairgrams(
          archive.getAs<ar::Boolean>("cross_column_pairgrams")) {
  for (const auto& num_col : archive.get("numerical_columns")->list()) {
    _numerical_columns.push_back(NumericalColumn(*num_col));
  }

  for (const auto& cat_col : archive.get("categorical_columns")->list()) {
    _categorical_columns.push_back(CategoricalColumn(*cat_col));
  }
}

ar::ConstArchivePtr NumericalColumn::toArchive() const {
  auto map = ar::Map::make();

  map->set("name", ar::str(name));
  map->set("min", ar::f32(_min));
  map->set("max", ar::f32(_max));
  map->set("binsize", ar::f32(_binsize));
  map->set("num_bins", ar::u64(_num_bins));
  map->set("salt", ar::u64(_salt));

  return map;
}

NumericalColumn::NumericalColumn(const ar::Archive& archive)
    : name(archive.str("name")),
      _min(archive.getAs<ar::F32>("min")),
      _max(archive.getAs<ar::F32>("max")),
      _binsize(archive.getAs<ar::F32>("binsize")),
      _num_bins(archive.u64("num_bins")),
      _salt(archive.u64("salt")) {}

ar::ConstArchivePtr CategoricalColumn::toArchive() const {
  auto map = ar::Map::make();

  map->set("name", ar::str(name));
  map->set("salt", ar::u64(_salt));

  return map;
}

CategoricalColumn::CategoricalColumn(const ar::Archive& archive)
    : name(archive.str("name")), _salt(archive.u64("salt")) {}

template <class Archive>
void Tabular::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _numerical_columns,
          _categorical_columns, _output_column, _cross_column_pairgrams);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::Tabular)