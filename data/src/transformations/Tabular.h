#pragma once

#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/TokenEncoding.h>

namespace thirdai::data {

struct NumericalColumn {
  explicit NumericalColumn(std::string _name, float _min, float _max,
                           uint32_t _num_bins)
      : name(std::move(_name)),
        min(_min),
        max(_max),
        binsize((max - min) / _num_bins),
        num_bins(_num_bins),
        salt(dataset::token_encoding::seededMurmurHash(name.data(),
                                                       name.size())) {}
  std::string name;
  float min;
  float max;
  float binsize;
  uint32_t num_bins;
  uint32_t salt;

  NumericalColumn() {}

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(name, min, max, binsize, num_bins, salt);
  }
};

struct CategoricalColumn {
  explicit CategoricalColumn(std::string _name)
      : name(std::move(_name)),
        salt(dataset::token_encoding::seededMurmurHash(name.data(),
                                                       name.size())) {}

  std::string name;
  uint32_t salt;

  CategoricalColumn() {}

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(name, salt);
  }
};

class Tabular final : public Transformation {
 public:
  Tabular(std::vector<NumericalColumn> numerical_columns,
          std::vector<CategoricalColumn> categorical_columns,
          std::string output_column, bool cross_column_pairgrams);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::vector<NumericalColumn> _numerical_columns;
  std::vector<CategoricalColumn> _categorical_columns;

  std::string _output_column;
  bool _cross_column_pairgrams;

  Tabular() {}

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data