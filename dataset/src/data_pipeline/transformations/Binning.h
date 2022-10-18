#pragma once

#include <dataset/src/data_pipeline/Transformation.h>
#include <dataset/src/data_pipeline/columns/VectorColumns.h>

namespace thirdai::dataset {

class BinningTransformation final : public Transformation {
 public:
  void apply(ColumnMap& columns) final {
    auto column = columns.getFloatValueColumn(_input_column);

    std::vector<uint32_t> binned_values(column->numRows());

#pragma omp parallel for default(none) shared(column, binned_values)
    for (uint64_t i = 0; i < column->numRows(); i++) {
      binned_values[i] = getBin((*column)[i]);
    }

    auto output_column = std::make_shared<VectorValueColumn<uint32_t>>(
        std::move(binned_values), _num_bins);

    columns.addColumn(_output_column, output_column);
  }

 private:
  uint32_t getBin(float value) const { return (value - _min_value) / _binsize; }

  std::string _input_column;
  std::string _output_column;

  float _min_value;
  float _binsize;
  uint32_t _num_bins;
};

}  // namespace thirdai::dataset