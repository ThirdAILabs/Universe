#include "Recurrence.h"
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <algorithm>
#include <cstddef>
#include <exception>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace thirdai::data {

static std::vector<size_t> offsets(const ArrayColumnBase<uint32_t>& column) {
  std::vector<size_t> offsets(column.numRows() + 1);
  offsets[0] = 0;
  for (uint32_t i = 0; i < column.numRows(); i++) {
    // +1 for EOS token.
    offsets[i + 1] = offsets[i] + column.row(i).size() + 1;
  }
  return offsets;
}

static std::vector<size_t> permutation(const std::vector<size_t>& offsets) {
  std::vector<size_t> permutation(offsets.back());

#pragma omp parallel for default(none) shared(permutation, offsets)
  for (uint32_t i = 0; i < offsets.size() - 1; i++) {
    for (uint32_t row_pos = offsets[i]; row_pos < offsets[i + 1]; ++row_pos) {
      permutation[row_pos] = i;
    }
  }

  return permutation;
}

static std::exception_ptr mismatchedRowSizeError(uint32_t source_row_size,
                                                 uint32_t target_row_size,
                                                 uint32_t row_number) {
  std::stringstream error_ss;
  error_ss << "Recurrence error: source is not the same size as target ("
           << source_row_size << " vs. " << target_row_size << ") in row "
           << row_number << ".";
  return std::make_exception_ptr(std::invalid_argument(error_ss.str()));
}

ColumnMap Recurrence::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto source_column = columns.getArrayColumn<uint32_t>(_source_input_column);
  auto target_column = columns.getArrayColumn<uint32_t>(_target_input_column);

  assertCorrectTargetInputDim(*target_column);

  std::vector<size_t> row_offsets = offsets(*source_column);
  const size_t total_num_rows = row_offsets.back();
  std::vector<std::vector<uint32_t>> unrolled_source_data(total_num_rows);
  std::vector<uint32_t> unrolled_target_data(total_num_rows);

  std::exception_ptr error;

#pragma omp parallel for default(none)                                      \
    shared(source_column, target_column, row_offsets, unrolled_source_data, \
           unrolled_target_data, error)
  for (uint32_t i = 0; i < source_column->numRows(); i++) {
    auto source_row = source_column->row(i);
    auto target_row = target_column->row(i);

    if (source_row.size() != target_row.size()) {
#pragma omp critical
      error = mismatchedRowSizeError(/* source_row_size= */ source_row.size(),
                                     /* target_row_size= */ target_row.size(),
                                     /* row_number= */ i);
    }

    try {
      const size_t offset = row_offsets[i];
      for (uint32_t row_pos = 0; row_pos < source_row.size(); ++row_pos) {
        // Simulate next token prediction by giving the model an array of tokens
        // up to the (row_pos - 1)th token.
        unrolled_source_data[offset + row_pos] =
            std::vector(source_row.begin(), source_row.begin() + row_pos);
        // Target is row_pos-th token; the next token to be predicted.
        unrolled_target_data[offset + row_pos] =
            offsetPosition(row_pos) * _target_vocab_size + target_row[row_pos];
      }
      unrolled_source_data[offset + source_row.size()] =
          std::vector(source_row.begin(), source_row.end());
      unrolled_target_data[offset + source_row.size()] =
          _target_vocab_size * _max_position_offset +
          offsetPosition(source_row.size());

    } catch (std::exception& e) {
#pragma omp critical
      error = std::make_exception_ptr(e);
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  auto unrolled_source_column = ArrayColumn<uint32_t>::make(
      std::move(unrolled_source_data), source_column->dim());
  auto unrolled_target_column = ValueColumn<uint32_t>::make(
      std::move(unrolled_target_data),
      // +1 for EOS.
      (_target_vocab_size + 1) * _max_position_offset);

  auto permutation_indices = permutation(row_offsets);
  columns = columns.permute(permutation_indices);
  columns.setColumn(_source_output_column, unrolled_source_column);
  columns.setColumn(_target_output_column, unrolled_target_column);
  return columns;
}

bool Recurrence::isEOS(uint32_t token) const {
  return token >= _target_vocab_size * _max_position_offset;
}

void Recurrence::assertCorrectTargetInputDim(
    const ArrayColumnBase<uint32_t>& target_column) const {
  if (!target_column.dim()) {
    throw std::invalid_argument(
        "Recurrence: Expected target column to have a dimension.");
  }
  if (*target_column.dim() != _target_vocab_size) {
    throw std::invalid_argument("Recurrence: Expected target vocab size " +
                                std::to_string(_target_vocab_size) +
                                " but received target column with dimension " +
                                std::to_string(*target_column.dim()));
  }
}

uint32_t Recurrence::offsetPosition(size_t position) const {
  return std::min(position, _max_position_offset - 1);
}
}  // namespace thirdai::data