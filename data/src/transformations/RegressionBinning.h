#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class RegressionBinning final : public Transformation {
 public:
  RegressionBinning(std::string input_column, std::string output_column,
                    float min, float max, size_t num_bins,
                    uint32_t correct_label_radius);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  uint32_t bin(float x) const;

  uint32_t labelStart(uint32_t center) const {
    return center < _correct_label_radius ? 0 : center - _correct_label_radius;
  }

  uint32_t labelEnd(uint32_t center) const {
    return std::min<uint32_t>(center + _correct_label_radius, _num_bins - 1);
  }

  std::string _input_column;
  std::string _output_column;

  float _min, _max, _binsize;
  size_t _num_bins;
  uint32_t _correct_label_radius;
};

}  // namespace thirdai::data