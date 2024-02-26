#include "DyadicContrastiveFeaturizer.h"
#include <data/src/ColumnMap.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/DyadicInterval.h>
#include <data/src/transformations/Transformation.h>
#include <exception>
#include <string>
#include <unordered_map>
#include <utility>

namespace thirdai::data {
DyadicContrastiveFeaturizer::DyadicContrastiveFeaturizer(
    std::string input_column_1, std::string input_column_2,
    std::optional<std::string> prompt_column, std::string label_column,
    std::string output_interval_prefix, size_t n_intervals, uint32_t n_classes,
    bool is_bidirectional)
    : _input_column_1(std::move(input_column_1)),
      _input_column_2(std::move(input_column_2)),
      _prompt_column(std::move(prompt_column)),
      _label_column(std::move(label_column)),
      _output_interval_prefix(std::move(output_interval_prefix)),
      _n_intervals(n_intervals),
      _n_classes(n_classes),
      _is_bidirectional(is_bidirectional) {}

std::pair<std::vector<std::vector<std::vector<uint32_t>>>,
          std::vector<std::vector<std::vector<uint32_t>>>>
DyadicContrastiveFeaturizer::featurizeColumnsDyadic(
    ArrayColumnBasePtr<uint32_t>& tokens) const {
  std::vector<std::vector<std::vector<uint32_t>>> intervals_from_end(
      _n_intervals);
  std::vector<std::vector<std::vector<uint32_t>>> intervals_from_start;
  for (size_t i = 0; i < _n_intervals; i++) {
    intervals_from_end.at(i).assign(tokens->numRows(), {});
  }
  if (_is_bidirectional) {
    intervals_from_start.resize(_n_intervals);

    for (size_t i = 0; i < _n_intervals; i++) {
      intervals_from_start.at(i).assign(tokens->numRows(), {});
    }
  }
  std::vector<std::vector<uint32_t>> prompt_inputs;

#pragma omp parallel for default(none)   \
    shared(tokens, intervals_from_start, \
           intervals_from_end) if (tokens->numRows() > 1)
  for (size_t i = 0; i < tokens->numRows(); i++) {
    auto row_tokens = tokens->row(i);

    for (size_t interval = 0; interval < _n_intervals; interval++) {
      size_t int_len = std::min<size_t>(row_tokens.size(), 1UL << interval);
      size_t int_start = row_tokens.size() - int_len;
      intervals_from_end[interval][i] =
          row_tokens.range(int_start, row_tokens.size());
      if (_is_bidirectional) {
        intervals_from_start[interval][i] = row_tokens.range(0, int_len);
      }
    }
  }
  return {intervals_from_start, intervals_from_end};
}

ColumnMap DyadicContrastiveFeaturizer::apply(ColumnMap columns,
                                             State& state) const {
  (void)state;
  auto tokens_1 = columns.getArrayColumn<uint32_t>(_input_column_1);
  auto tokens_2 = columns.getArrayColumn<uint32_t>(_input_column_2);
  auto [interval_from_start_1, interval_from_end_1] =
      featurizeColumnsDyadic(tokens_1);
  auto [interval_from_start_2, interval_from_end_2] =
      featurizeColumnsDyadic(tokens_2);
  std::unordered_map<std::string, ColumnPtr> output_columns;

  for (size_t interval = 0; interval < _n_intervals; interval++) {
    std::string name =
        _output_interval_prefix + "from_end_" + std::to_string(1 << interval);

    output_columns[name + "_1"] = ArrayColumn<uint32_t>::make(
        std::move(interval_from_end_1[interval]), tokens_1->dim());
    output_columns[name + "_2"] = ArrayColumn<uint32_t>::make(
        std::move(interval_from_end_2[interval]), tokens_2->dim());
  }

  if (_is_bidirectional) {
    for (size_t interval = 0; interval < _n_intervals; interval++) {
      std::string name = _output_interval_prefix + "from_start_" +
                         std::to_string(1 << interval);

      output_columns[name + "_1"] = ArrayColumn<uint32_t>::make(
          std::move(interval_from_start_1[interval]), tokens_1->dim());
      output_columns[name + "_2"] = ArrayColumn<uint32_t>::make(
          std::move(interval_from_start_2[interval]), tokens_2->dim());
    }
  }
  if (_prompt_column) {
    output_columns[*_prompt_column] =
        columns.getArrayColumn<uint32_t>(*_prompt_column);
  }

  return ColumnMap(output_columns);
}

ar::ConstArchivePtr DyadicContrastiveFeaturizer::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  if (_prompt_column) {
    map->set("prompt_column", ar::str(*_prompt_column));
  }
  map->set("input_column_1", ar::str(_input_column_1));
  map->set("input_column_2", ar::str(_input_column_2));
  map->set("label_column", ar::str(_label_column));

  map->set("n_classes", ar::u64(_n_classes));

  map->set("output_interval_prefix", ar::str(_output_interval_prefix));

  map->set("is_bidirectional", ar::boolean(_is_bidirectional));
  map->set("n_intervals", ar::u64(_n_intervals));

  return map;
}

ColumnMap DyadicContrastiveFeaturizer::inferenceFeaturization(
    ColumnMap columns) const {
  data::State state;
  return apply(std::move(columns), state);
}

}  // namespace thirdai::data