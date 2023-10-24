#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class DyadicIntervalClassification : public Transformation {
 public:
  DyadicIntervalClassification(std::string input_column,
                               std::optional<std::string> prompt_column,
                               std::string label_column,
                               std::string output_interval_prefix,
                               size_t n_intervals, uint32_t n_classes);
  ColumnMap apply(ColumnMap columns, State& state) const final;

  ColumnMap inferenceFeaturization(ColumnMap columns) const;

 private:
  std::string _input_column;
  std::optional<std::string> _prompt_column;
  std::string _label_column;
  std::string _output_interval_prefix;
  size_t _n_intervals;
  uint32_t _n_classes;
};

}  // namespace thirdai::data