#pragma once

#include <data/src/transformations/Transformation.h>
#include <string>
#include <vector>

namespace thirdai::data {

/**
 * Unrolls a sequence into a separate row for each time step.
 * source_input_column and target_input_column are representations of the
 * same sequences. This gives us the flexibility to featurize source and
 * target tokens differently, e.g. use hash-based position encoding for the
 * source sequence and use offset-based position encoding for the target
 * sequence.
 */
class UnrollSequence final : public Transformation {
 public:
  UnrollSequence(std::string source_input_column,
                 std::string target_input_column,
                 std::string source_output_column,
                 std::string target_output_column)
      : _source_input_column(std::move(source_input_column)),
        _target_input_column(std::move(target_input_column)),
        _source_output_column(std::move(source_output_column)),
        _target_output_column(std::move(target_output_column)) {}

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _source_input_column;
  std::string _target_input_column;
  std::string _source_output_column;
  std::string _target_output_column;
};

}  // namespace thirdai::data