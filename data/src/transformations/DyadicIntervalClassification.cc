#include "DyadicIntervalClassification.h"
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <exception>
#include <string>
#include <unordered_map>
#include <utility>

namespace thirdai::data {
DyadicIntervalClassification::DyadicIntervalClassification(
    std::string input_column, std::optional<std::string> prompt_column,
    std::string label_column, std::string output_interval_prefix,
    size_t n_intervals, uint32_t n_classes)
    : _input_column(std::move(input_column)),
      _prompt_column(std::move(prompt_column)),
      _label_column(std::move(label_column)),
      _output_interval_prefix(std::move(output_interval_prefix)),
      _n_intervals(n_intervals),
      _n_classes(n_classes) {}

ColumnMap DyadicIntervalClassification::apply(ColumnMap columns,
                                              State& state) const {
  (void)state;
  auto tokens = columns.getArrayColumn<uint32_t>(_input_column);
  auto labels = columns.getValueColumn<uint32_t>(_label_column);
  std::vector<std::vector<std::vector<uint32_t>>> intervals(_n_intervals);
  std::vector<uint32_t> targets(tokens->numRows());
  for (size_t i = 0; i < _n_intervals; i++) {
    intervals.at(i).assign(columns.numRows(), {});
  }

  std::vector<std::vector<uint32_t>> prompt_inputs;
  ArrayColumnBasePtr<uint32_t> prompts;
  if (_prompt_column) {
    prompt_inputs.resize(tokens->numRows());
    prompts = columns.getArrayColumn<uint32_t>(*_prompt_column);
  }
#pragma omp parallel for default(none)                  \
    shared(tokens, intervals, labels, targets, prompts, \
           prompt_inputs) if (tokens->numRows() > 1)
  for (size_t i = 0; i < tokens->numRows(); i++) {
    auto row_tokens = tokens->row(i);

    for (size_t interval = 0; interval < _n_intervals; interval++) {
      size_t int_len = std::min<size_t>(row_tokens.size(), 1UL << interval);
      size_t int_start = row_tokens.size() - int_len;
      intervals[interval][i] = row_tokens.range(int_start, row_tokens.size());
    }
    targets[i] = labels->value(i);
    if (_prompt_column) {
      auto prompt = prompts->row(i);
      prompt_inputs[i] = {prompt.begin(), prompt.end()};
    }
  }
  std::unordered_map<std::string, ColumnPtr> output_columns;
  for (size_t interval = 0; interval < _n_intervals; interval++) {
    std::string name =
        _output_interval_prefix + "from_end_" + std::to_string(1 << interval);

    output_columns[name] = ArrayColumn<uint32_t>::make(
        std::move(intervals[interval]), tokens->dim());
  }
  output_columns[_label_column] =
      ValueColumn<uint32_t>::make(std::move(targets), _n_classes);
  if (_prompt_column) {
    output_columns[*_prompt_column] =
        ArrayColumn<uint32_t>::make(std::move(prompt_inputs), prompts->dim());
  }

  return ColumnMap(output_columns);
}

}  // namespace thirdai::data
