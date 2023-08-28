#include "RegressionBinning.h"
#include <data/src/columns/ArrayColumns.h>
#include <proto/transformations.pb.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace thirdai::data {

RegressionBinning::RegressionBinning(std::string input_column,
                                     std::string output_column, float min,
                                     float max, size_t num_bins,
                                     uint32_t correct_label_radius)
    : _input_column(std::move(input_column)),
      _output_column(std::move(output_column)),
      _min(min),
      _max(max),
      _binsize((max - min) / num_bins),
      _num_bins(num_bins),
      _correct_label_radius(correct_label_radius) {
  if (min >= max) {
    throw std::invalid_argument("min must be < max in RegressionBinning.");
  }
}

RegressionBinning::RegressionBinning(
    const proto::data::RegressionBinning& regression)
    : _input_column(regression.input_column()),
      _output_column(regression.output_column()),
      _min(regression.min()),
      _max(regression.max()),
      _binsize(regression.binsize()),
      _num_bins(regression.num_bins()),
      _correct_label_radius(regression.correct_label_radius()) {}

ColumnMap RegressionBinning::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto float_col = columns.getValueColumn<float>(_input_column);

  std::vector<std::vector<uint32_t>> labels(float_col->numRows());

#pragma omp parallel for default(none) shared(float_col, labels)
  for (size_t i = 0; i < float_col->numRows(); i++) {
    uint32_t center = bin(float_col->value(i));

    uint32_t label_start = labelStart(center);

    std::vector<uint32_t> sample_label(labelEnd(center) - label_start + 1);
    std::iota(sample_label.begin(), sample_label.end(), label_start);

    labels[i] = std::move(sample_label);
  }

  auto labels_col = ArrayColumn<uint32_t>::make(std::move(labels), _num_bins);
  columns.setColumn(_output_column, labels_col);

  return columns;
}

uint32_t RegressionBinning::bin(float x) const {
  uint32_t bin = (std::clamp(x, _min, _max) - _min) / _binsize;

  // Because we clamp to range [min, max], we could theorically reach the
  // value of dim since max = dim * binsize + min.
  bin = std::min<uint32_t>(bin, _num_bins - 1);

  return bin;
}

proto::data::Transformation* RegressionBinning::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* regression = transformation->mutable_regression_binning();

  regression->set_input_column(_input_column);
  regression->set_output_column(_output_column);
  regression->set_min(_min);
  regression->set_max(_max);
  regression->set_binsize(_binsize);
  regression->set_num_bins(_num_bins);
  regression->set_correct_label_radius(_correct_label_radius);

  return transformation;
}

}  // namespace thirdai::data
