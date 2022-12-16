#include "Binning.h"

namespace thirdai::data {

void BinningTransformation::apply(ColumnMap& columns) {
  auto column = columns.getDenseFeatureColumn(_input_column_name);

  std::vector<uint32_t> binned_values(column->numRows());

  std::optional<float> invalid_value = std::nullopt;
#pragma omp parallel for default(none) \
    shared(column, binned_values, invalid_value)
  for (uint64_t i = 0; i < column->numRows(); i++) {
    if (auto bin = getBin((*column)[i])) {
      binned_values[i] = *bin;
    } else {
#pragma omp critical
      invalid_value = (*column)[i];
    }
  }

  if (invalid_value) {
    throw std::invalid_argument(
        "Cannot bin value " + std::to_string(invalid_value.value()) +
        ". Expected values in range [" + std::to_string(_inclusive_min_value) +
        ", " + std::to_string(_exclusive_max_value) + ").");
  }

  auto output_column = std::make_shared<columns::CppTokenColumn>(
      std::move(binned_values), _num_bins);

  columns.setColumn(_output_column_name, output_column);
}

std::optional<uint32_t> BinningTransformation::getBin(float value) const {
  if (value >= _exclusive_max_value || value < _inclusive_min_value) {
    return std::nullopt;
  }
  return (value - _inclusive_min_value) / _binsize;
}

}  // namespace thirdai::data