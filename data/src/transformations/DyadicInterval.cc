#include "DyadicInterval.h"
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <string>
#include <unordered_map>

namespace thirdai::data {

ColumnMap DyadicInterval::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto texts = columns.getArrayColumn<uint32_t>(_input_column);

  std::vector<size_t> sample_offsets = computeOffsets(texts);

  std::vector<std::vector<std::vector<uint32_t>>> intervals(_n_intervals);
  for (size_t i = 0; i < _n_intervals; i++) {
    intervals.at(i).assign(sample_offsets.back(), {});
  }
  std::vector<uint32_t> labels(sample_offsets.back());

  size_t chunk_size = (1 << _n_intervals) + 1;

  for (size_t i = 0; i < texts->numRows(); i++) {
    auto tokens = texts->row(i);

    size_t sample_offset = sample_offsets[i];

    for (size_t start = 0; start < tokens.size(); start += chunk_size) {
      size_t end = std::min(start + chunk_size, tokens.size());
      for (size_t target = start + 1; target <= end; target++) {
        for (size_t interval = 0; interval < _n_intervals; interval++) {
          size_t int_len = std::min(target - start, 1UL << (interval + 1));
          size_t int_start = target - int_len;
          intervals[interval][sample_offset] = tokens.range(int_start, target);
        }

        labels[sample_offset] = tokens[target];

        sample_offset++;
      }
    }
  }

  std::unordered_map<std::string, ColumnPtr> output_columns;

  for (size_t interval = 0; interval < _n_intervals; interval++) {
    std::string name =
        _output_interval_prefix + std::to_string(1 << (interval + 1));

    output_columns[name] = ArrayColumn<uint32_t>::make(
        std::move(intervals[interval]), _vocab_size);
  }

  output_columns[_target_column] =
      ValueColumn<uint32_t>::make(std::move(labels), _vocab_size);

  return ColumnMap(output_columns);
}

std::vector<size_t> DyadicInterval::computeOffsets(
    const ArrayColumnBasePtr<uint32_t>& texts) {
  std::vector<size_t> offsets(texts->numRows() + 1);
  offsets[0] = 0;

  for (size_t i = 0; i < texts->numRows(); i++) {
    offsets[i + 1] = offsets[i] + texts->row(i).size();
  }
  return offsets;
}

}  // namespace thirdai::data