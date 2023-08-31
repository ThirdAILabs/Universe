#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class DyadicInterval final : public Transformation {
 public:
  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  static std::vector<size_t> computeOffsets(
      const ArrayColumnBasePtr<uint32_t>& texts);

  std::string _input_column;
  std::string _output_interval_prefix;
  std::string _target_column;

  size_t _n_intervals;
  size_t _vocab_size;
};

}  // namespace thirdai::data