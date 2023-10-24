#pragma once

#include <data/src/transformations/Transformation.h>
#include <proto/binning.pb.h>
#include <memory>
#include <utility>

namespace thirdai::data {

/**
 * This transformation is for converting regression tasks to categorical tasks.
 * It works by binning the decimal value and then creating categorical labels
 * for the bin and the surrounding bins. The number of surrounding bins is
 * determined by the parameter 'correct_label_radius' and it is to control how
 * much the model is penalized for predicting close the correct bin/value, but
 * not the exact right bin. For example if the target bin is 10, and the radius
 * is 3, then the labels will be [7, 8, 9, 10, 11, 12, 13].
 */
class RegressionBinning final : public Transformation {
 public:
  RegressionBinning(std::string input_column, std::string output_column,
                    float min, float max, size_t num_bins,
                    uint32_t correct_label_radius);

  static std::shared_ptr<RegressionBinning> make(
      std::string input_column, std::string output_column, float min, float max,
      size_t num_bins, uint32_t correct_label_radius) {
    return std::make_shared<RegressionBinning>(
        std::move(input_column), std::move(output_column), min, max, num_bins,
        correct_label_radius);
  }

  explicit RegressionBinning(const proto::data::RegressionBinning& regression);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  proto::data::Transformation* toProto() const final;

  float unbin(uint32_t category) const {
    return _min + category * _binsize + (_binsize / 2);
  }

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