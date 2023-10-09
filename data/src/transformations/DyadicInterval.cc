#include "DyadicInterval.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <exception>
#include <string>
#include <unordered_map>

namespace thirdai::data {

DyadicInterval::DyadicInterval(std::string input_column,
                               std::optional<std::string> prompt_column,
                               std::string output_interval_prefix,
                               std::string target_column, size_t n_intervals,
                               bool is_bidirectional)
    : _prompt_column(std::move(prompt_column)),
      _input_column(std::move(input_column)),
      _output_interval_prefix(std::move(output_interval_prefix)),
      _target_column(std::move(target_column)),
      _is_bidirectional(is_bidirectional),
      _n_intervals(n_intervals) {}

ColumnMap DyadicInterval::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto texts = columns.getArrayColumn<uint32_t>(_input_column);

  ArrayColumnBasePtr<uint32_t> prompts;

  size_t chunk_size = (1UL << (_n_intervals - 1)) + 1;

  std::vector<size_t> sample_offsets = computeOffsets(texts, chunk_size);

  std::vector<std::vector<std::vector<uint32_t>>> intervals(_n_intervals);
  for (size_t i = 0; i < _n_intervals; i++) {
    intervals.at(i).assign(sample_offsets.back(), {});
  }
  if (_is_bidirectional) {
    intervals.resize(2 * _n_intervals);

    for (size_t i = 0; i < _n_intervals; i++) {
      intervals.at(_n_intervals + i).assign(sample_offsets.back(), {});
    }
  }
  std::vector<uint32_t> targets(sample_offsets.back());
  std::vector<std::vector<uint32_t>> prompt_inputs;
  if (_prompt_column) {
    prompt_inputs.resize(sample_offsets.back());
    prompts = columns.getArrayColumn<uint32_t>(*_prompt_column);
  }

  std::exception_ptr error;

#pragma omp parallel for default(none)                                \
    shared(texts, sample_offsets, intervals, _prompt_column, prompts, \
           prompt_inputs, targets, chunk_size, error)
  for (size_t i = 0; i < texts->numRows(); i++) {
    try {
      auto tokens = texts->row(i);

      size_t sample_offset = sample_offsets[i];

      for (size_t start = 0; start < tokens.size(); start += chunk_size) {
        size_t end = std::min(start + chunk_size, tokens.size());
        for (size_t target = start + 1; target < end; target++) {
          for (size_t interval = 0; interval < _n_intervals; interval++) {
            size_t int_len = std::min<size_t>(target - start, 1UL << interval);
            size_t int_start = target - int_len;
            intervals[interval][sample_offset] =
                tokens.range(int_start, target);
            if (_is_bidirectional) {
              assert(intervals.size() == 2 * _n_intervals);
              size_t int_end = start + int_len;
              intervals[_n_intervals + interval][sample_offset] =
                  tokens.range(start, int_end);
            }
          }

          targets[sample_offset] = tokens[target];
          if (_prompt_column) {
            auto prompt = prompts->row(i);
            prompt_inputs[sample_offset] = prompt.range(0, prompt.size());
          }

          sample_offset++;
        }
      }
    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  std::unordered_map<std::string, ColumnPtr> output_columns;

  for (size_t interval = 0; interval < _n_intervals; interval++) {
    std::string name = _output_interval_prefix + std::to_string(1 << interval);

    output_columns[name] = ArrayColumn<uint32_t>::make(
        std::move(intervals[interval]), texts->dim());
  }

  if (_is_bidirectional) {
    assert(intervals.size() == 2 * _n_intervals);
    for (size_t interval = 0; interval < _n_intervals; interval++) {
      std::string name =
          "rev_" + _output_interval_prefix + std::to_string(1 << interval);

      output_columns[name] = ArrayColumn<uint32_t>::make(
          std::move(intervals[_n_intervals + interval]), texts->dim());
    }
  }

  output_columns[_target_column] =
      ValueColumn<uint32_t>::make(std::move(targets), texts->dim());

  if (_prompt_column) {
    output_columns[*_prompt_column] =
        ArrayColumn<uint32_t>::make(std::move(prompt_inputs), prompts->dim());
  }

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

ColumnMap DyadicInterval::inferenceFeaturization(ColumnMap columns) const {
  auto tokens = columns.getArrayColumn<uint32_t>(_input_column);

  std::vector<std::vector<std::vector<uint32_t>>> intervals(_n_intervals);
  for (size_t i = 0; i < _n_intervals; i++) {
    intervals.at(i).assign(columns.numRows(), {});
  }
  if (_is_bidirectional) {
    intervals.resize(2 * _n_intervals);

    for (size_t i = 0; i < _n_intervals; i++) {
      intervals.at(_n_intervals + i).assign(columns.numRows(), {});
    }
  }

#pragma omp parallel for default(none) \
    shared(tokens, intervals) if (tokens->numRows() > 1)
  for (size_t i = 0; i < tokens->numRows(); i++) {
    auto row_tokens = tokens->row(i);

    for (size_t interval = 0; interval < _n_intervals; interval++) {
      size_t int_len = std::min<size_t>(row_tokens.size(), 1UL << interval);
      size_t int_start = row_tokens.size() - int_len;
      intervals[interval][i] = row_tokens.range(int_start, row_tokens.size());
      if (_is_bidirectional) {
        assert(intervals.size() == 2 * _n_intervals);
        intervals[_n_intervals + interval][i] = row_tokens.range(0, int_len);
      }
    }
  }

  for (size_t interval = 0; interval < _n_intervals; interval++) {
    std::string name = _output_interval_prefix + std::to_string(1 << interval);

    columns.setColumn(name, ArrayColumn<uint32_t>::make(
                                std::move(intervals[interval]), tokens->dim()));
  }

  if (_is_bidirectional) {
    for (size_t interval = 0; interval < _n_intervals; interval++) {
      std::string name =
          "rev_" + _output_interval_prefix + std::to_string(1 << interval);

      columns.setColumn(name, ArrayColumn<uint32_t>::make(
                                  std::move(intervals[interval + _n_intervals]),
                                  tokens->dim()));
    }
  }

  return columns;
}

template void DyadicInterval::serialize(cereal::BinaryInputArchive&);
template void DyadicInterval::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void DyadicInterval::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column,
          _output_interval_prefix, _target_column, _n_intervals,
          _is_bidirectional, _prompt_column);
}

}  // namespace thirdai::data