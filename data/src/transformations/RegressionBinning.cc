#include "RegressionBinning.h"
#include <data/src/columns/ArrayColumns.h>
#include <algorithm>
#include <numeric>

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
      _correct_label_radius(correct_label_radius) {}

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

}  // namespace thirdai::data