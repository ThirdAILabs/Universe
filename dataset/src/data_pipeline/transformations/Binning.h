#pragma once

#include <dataset/src/data_pipeline/Transformation.h>
#include <dataset/src/data_pipeline/columns/VectorColumns.h>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

// Bins a dense float column into categorical sparse values. If the input column
// is the same as the output column then that column will be replaced in the
// column map.
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

  void apply(ColumnMap& columns) final;

 private:
  std::optional<uint32_t> getBin(float value) const;

  std::string _input_column_name;
  std::string _output_column_name;

  float _inclusive_min_value;
  float _exclusive_max_value;
  float _binsize;
  uint32_t _num_bins;
};

}  // namespace thirdai::dataset