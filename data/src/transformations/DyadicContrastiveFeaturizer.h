#pragma once


#include "DyadicInterval.h"
#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/columns/ArrayColumns.h>

namespace thirdai::data {

class DyadicContrastiveFeaturizer : public Transformation {
 public:
  DyadicContrastiveFeaturizer(std::string input_column_1,
                              std::string input_column_2,
                              std::optional<std::string> prompt_column,
                              std::string label_column,
                              std::string output_interval_prefix,
                              size_t n_intervals, uint32_t n_classes,
                              bool is_bidirectional = false);
  ColumnMap apply(ColumnMap columns, State& state) const final;

  ColumnMap inferenceFeaturization(ColumnMap columns) const;

  std::pair<std::vector<std::vector<std::vector<uint32_t>>>,
            std::vector<std::vector<std::vector<uint32_t>>>>
  featurizeColumnsDyadic(ArrayColumnBasePtr<uint32_t> &tokens) const;

 private:
  std::string _input_column_1;
  std::string _input_column_2;
  std::optional<std::string> _prompt_column;
  std::string _label_column;
  std::string _output_interval_prefix;
  size_t _n_intervals;
  uint32_t _n_classes;
  bool _is_bidirectional;
};

}  // namespace thirdai::data