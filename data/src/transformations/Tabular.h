#pragma once

#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <proto/tabular.pb.h>
#include <proto/transformations.pb.h>

namespace thirdai::data {

struct NumericalColumn {
  explicit NumericalColumn(std::string _name, float min, float max,
                           uint32_t num_bins)
      : name(std::move(_name)),
        _min(min),
        _max(max),
        _binsize((max - min) / num_bins),
        _num_bins(num_bins),
        _salt(dataset::token_encoding::seededMurmurHash(name.data(),
                                                        name.size())) {}

  explicit NumericalColumn(const proto::data::NumericalColumn& num_col);

  inline uint32_t encode(const std::string& val) const;

  proto::data::NumericalColumn* toProto() const;

  std::string name;

 private:
  float _min;
  float _max;
  float _binsize;
  uint32_t _num_bins;
  uint32_t _salt;
};

struct CategoricalColumn {
  explicit CategoricalColumn(std::string _name)
      : name(std::move(_name)),
        _salt(dataset::token_encoding::seededMurmurHash(name.data(),
                                                        name.size())) {}

  explicit CategoricalColumn(const proto::data::CategoricalColumn& cat_col);

  inline uint32_t encode(const std::string& val) const;

  proto::data::CategoricalColumn* toProto() const;

  std::string name;

 private:
  uint32_t _salt;
};

class Tabular final : public Transformation {
 public:
  Tabular(std::vector<NumericalColumn> numerical_columns,
          std::vector<CategoricalColumn> categorical_columns,
          std::string output_column, bool cross_column_pairgrams);

  explicit Tabular(const proto::data::Tabular& tabular);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

  const auto& numericalColumns() const { return _numerical_columns; }

  const auto& categoricalColumns() const { return _categorical_columns; }

 private:
  std::vector<NumericalColumn> _numerical_columns;
  std::vector<CategoricalColumn> _categorical_columns;

  std::string _output_column;
  bool _cross_column_pairgrams;
};

}  // namespace thirdai::data