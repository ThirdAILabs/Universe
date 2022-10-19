#pragma once

#include <dataset/src/data_pipeline/Transformation.h>
#include <dataset/src/data_pipeline/columns/VectorColumns.h>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

class BinningTransformation final : public Transformation {
 public:
  BinningTransformation(std::string input_column_name,
                        std::string output_column_name,
                        float inclusive_min_value, float exclusive_max_value,
                        uint32_t num_bins)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)),
        _inclusive_min_value(inclusive_min_value),
        _exclusive_max_value(exclusive_max_value),
        _binsize((exclusive_max_value - inclusive_min_value) / num_bins),
        _num_bins(num_bins) {}

  void apply(ColumnMap& columns) final {
    auto column = columns.getFloatValueColumn(_input_column_name);

    std::vector<uint32_t> binned_values(column->numRows());

    std::optional<float> invalid_value = std::nullopt;
#pragma omp parallel for default(none) \
    shared(column, binned_values, invalid_value)
    for (uint64_t i = 0; i < column->numRows(); i++) {
      if (auto bin = getBin((*column)[i])) {
        binned_values[i] = bin.value();
      } else {
#pragma omp critical
        invalid_value = (*column)[i];
      }
    }

    if (invalid_value) {
      throw std::invalid_argument("Cannot bin value " +
                                  std::to_string(invalid_value.value()) +
                                  ". Expected values in range [" +
                                  std::to_string(_inclusive_min_value) + ", " +
                                  std::to_string(_exclusive_max_value) + ").");
    }

    auto output_column = std::make_shared<VectorValueColumn<uint32_t>>(
        std::move(binned_values), _num_bins);

    columns.addColumn(_output_column_name, output_column);
  }

 private:
  std::optional<uint32_t> getBin(float value) const {
    if (value >= _exclusive_max_value || value < _inclusive_min_value) {
      return std::nullopt;
    }
    return (value - _inclusive_min_value) / _binsize;
  }

  std::string _input_column_name;
  std::string _output_column_name;

  float _inclusive_min_value;
  float _exclusive_max_value;
  float _binsize;
  uint32_t _num_bins;
};

}  // namespace thirdai::dataset