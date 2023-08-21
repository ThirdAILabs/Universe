#pragma once

#include <data/src/columns/Column.h>
#include <data/src/transformations/Transformation.h>
#include <algorithm>
#include <cstddef>
#include <optional>
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
 *
 * For the source sequence [a, b, c, d] and target sequence [A, B, C, D] it will
 * generate the samples:
 * []        -> A
 * [a]       -> B
 * [a, b]    -> C
 * [a, b, c] -> D
 *
 * Left of "->" is source_output_column
 * Right of "->" is target_output_column
 */
class UnrollSequence final : public Transformation {
 public:
  UnrollSequence(std::string source_input_column,
                 std::string target_input_column,
                 std::string source_output_column,
                 std::string target_output_column, size_t target_vocab_size,
                 size_t max_position_offset)
      : _source_input_column(std::move(source_input_column)),
        _target_input_column(std::move(target_input_column)),
        _source_output_column(std::move(source_output_column)),
        _target_output_column(std::move(target_output_column)),
        _target_vocab_size(target_vocab_size),
        _max_position_offset(max_position_offset) {}

  ColumnMap apply(ColumnMap columns, State& state) const final;

  bool isEOS(uint32_t token) const;

 private:
  void assertCorrectTargetInputDim(
      const ArrayColumnBase<uint32_t>& target_column) const;

  uint32_t tokenOffset(size_t position) const;

  std::string _source_input_column;
  std::string _target_input_column;
  std::string _source_output_column;
  std::string _target_output_column;
  size_t _target_vocab_size;
  size_t _max_position_offset;
};

}  // namespace thirdai::data