#include "DyadicInterval.h"
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <string>
#include <unordered_map>

namespace thirdai::data {

DyadicInterval::DyadicInterval(std::string input_column,
                               std::string output_interval_prefix,
                               std::string target_column, size_t n_intervals)
    : _input_column(std::move(input_column)),
      _output_interval_prefix(std::move(output_interval_prefix)),
      _target_column(std::move(target_column)),
      _n_intervals(n_intervals) {}

ColumnMap DyadicInterval::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto texts = columns.getArrayColumn<uint32_t>(_input_column);

  size_t chunk_size = (1 << (_n_intervals - 1)) + 1;

  std::vector<size_t> sample_offsets = computeOffsets(texts, chunk_size);

  std::vector<std::vector<std::vector<uint32_t>>> intervals(_n_intervals);
  for (size_t i = 0; i < _n_intervals; i++) {
    intervals.at(i).assign(sample_offsets.back(), {});
  }
  std::vector<uint32_t> targets(sample_offsets.back());

#pragma omp parallel for default(none) \
    shared(texts, sample_offsets, intervals, targets, chunk_size)
  for (size_t i = 0; i < texts->numRows(); i++) {
    auto tokens = texts->row(i);

    size_t sample_offset = sample_offsets[i];

    for (size_t start = 0; start < tokens.size(); start += chunk_size) {
      size_t end = std::min(start + chunk_size, tokens.size());
      for (size_t target = start + 1; target < end; target++) {
        for (size_t interval = 0; interval < _n_intervals; interval++) {
          size_t int_len = std::min(target - start, 1UL << interval);
          size_t int_start = target - int_len;
          intervals[interval][sample_offset] = tokens.range(int_start, target);
        }

        targets[sample_offset] = tokens[target];

        sample_offset++;
      }
    }
  }

  std::unordered_map<std::string, ColumnPtr> output_columns;

  for (size_t interval = 0; interval < _n_intervals; interval++) {
    std::string name = _output_interval_prefix + std::to_string(1 << interval);

    output_columns[name] = ArrayColumn<uint32_t>::make(
        std::move(intervals[interval]), texts->dim());
  }

  output_columns[_target_column] =
      ValueColumn<uint32_t>::make(std::move(targets), texts->dim());

  return ColumnMap(output_columns);
}

std::vector<size_t> DyadicInterval::computeOffsets(
    const ArrayColumnBasePtr<uint32_t>& texts, size_t chunk_size) {
  std::vector<size_t> offsets(texts->numRows() + 1);
  offsets[0] = 0;

  for (size_t i = 0; i < texts->numRows(); i++) {
    size_t text_len = texts->row(i).size();
    size_t n_chunks = (text_len + chunk_size - 1) / chunk_size;
    // Since we always with at least 1 token as context, hence a chunk of size 4
    // only yields 3 samples.
    offsets[i + 1] = offsets[i] + (text_len - n_chunks);
  }
  return offsets;
}

}  // namespace thirdai::data