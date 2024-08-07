#pragma once

#include <archive/src/Archive.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/TokenEncoding.h>

namespace thirdai::data {

inline float parseFloat(const std::string& str) {
  if (str.empty()) {
    return 0.0;
  }
  return std::stof(str);
}

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

  explicit NumericalColumn(const ar::Archive& archive);

  inline uint32_t encode(const std::string& str_val) const {
    float val;
    try {
      val = parseFloat(str_val);
    } catch (...) {
      std::stringstream error;
      error << "Cannot cast '" << str_val << "' to a float";
      throw std::invalid_argument(error.str());
    }

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

  std::string name;

  NumericalColumn() {}

  ar::ConstArchivePtr toArchive() const;

 private:
  float _min;
  float _max;
  float _binsize;
  uint32_t _num_bins;
  uint32_t _salt;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(name, _min, _max, _binsize, _num_bins, _salt);
  }
};

struct CategoricalColumn {
  explicit CategoricalColumn(std::string _name)
      : name(std::move(_name)),
        _salt(dataset::token_encoding::seededMurmurHash(name.data(),
                                                        name.size())) {}

  explicit CategoricalColumn(const ar::Archive& archive);

  inline uint32_t encode(const std::string& val) const;

  std::string name;

  CategoricalColumn() {}

  ar::ConstArchivePtr toArchive() const;

 private:
  uint32_t _salt;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(name, _salt);
  }
};

class Tabular final : public Transformation {
 public:
  Tabular(std::vector<NumericalColumn> numerical_columns,
          std::vector<CategoricalColumn> categorical_columns,
          std::string output_column, bool cross_column_pairgrams);

  explicit Tabular(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  const auto& numericalColumns() const { return _numerical_columns; }

  const auto& categoricalColumns() const { return _categorical_columns; }

  static std::string type() { return "tabular"; }

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