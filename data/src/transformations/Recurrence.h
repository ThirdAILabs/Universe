#pragma once

#include <data/src/columns/Column.h>
#include <data/src/transformations/Transformation.h>
#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace thirdai::data {

/**
 * This class transforms the input data for training a recurrent sequence
 * prediction model. Given a source and target sequence, it will do three
 * things:
 * 1. Adds an EOS token to the target sequence
 * 2. Encodes the position of each token in the target sequence (up to
 * max_positions)
 * 3. Unrolls the sequence into a separate row for each time step.
 *
 * For the source sequence [a, b, c, d], target sequence [A, B, C, D], and
 * max_positions = 3, it will generate the samples:
 * []           -> A0 (A with encoding of 0th position)
 * [a]          -> B1
 * [a, b]       -> C2
 * [a, b, c]    -> D2 (max_positions = 3, so maximum encoded position is 2)
 * [a, b, c, d] -> EOS2 (end of sequence token, with encoding of position = 2)
 *
 * Left of "->" is source_output_column
 * Right of "->" is target_output_column
 *
 * Note that source_input_column and target_input_column are representations of
 * the same sequences. This gives us the flexibility to featurize the source
 * tokens in a different way than the target tokens.
 */
class Recurrence final : public Transformation {
 public:
  Recurrence(std::string source_input_column, std::string target_input_column,
             std::string source_output_column, std::string target_output_column,
             size_t target_vocab_size, size_t max_sequence_length)
      : _source_input_column(std::move(source_input_column)),
        _target_input_column(std::move(target_input_column)),
        _source_output_column(std::move(source_output_column)),
        _target_output_column(std::move(target_output_column)),
        _target_vocab_size(target_vocab_size),
        _max_seq_len(max_sequence_length) {}

  ColumnMap apply(ColumnMap columns, State& state) const final;

  bool isEOS(uint32_t token) const;

  inline constexpr size_t totalVocabSize() const {
    // +1 for EOS.
    return _target_vocab_size + 1;
  }

  inline constexpr size_t outputDim() const { return totalVocabSize(); }

  std::pair<uint32_t, uint32_t> rangeForStep(uint32_t step) const {
    (void)step;
    return {0, totalVocabSize()};
  }

  static uint32_t toTargetInputToken(uint32_t target_output_token) {
    return target_output_token;
  }

 private:
  size_t effectiveSize(const RowView<uint32_t>& row) const;

  std::vector<size_t> offsets(const ArrayColumnBase<uint32_t>& column) const;

  void assertCorrectTargetInputDim(
      const ArrayColumnBase<uint32_t>& target_column) const;

  inline constexpr uint32_t EOS() const { return _target_vocab_size; }

  std::string _source_input_column;
  std::string _target_input_column;
  std::string _source_output_column;
  std::string _target_output_column;
  size_t _target_vocab_size;
  size_t _max_seq_len;

  Recurrence() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data