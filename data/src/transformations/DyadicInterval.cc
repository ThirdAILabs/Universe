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
                               std::optional<std::string> context_column,
                               std::optional<std::string> prompt_column,
                               std::string output_interval_prefix,
                               std::string target_column, size_t n_intervals,
                               bool is_bidirectional)
    : _input_column(std::move(input_column)),
      _context_column(std::move(context_column)),
      _prompt_column(std::move(prompt_column)),
      _output_interval_prefix(std::move(output_interval_prefix)),
      _target_column(std::move(target_column)),
      _is_bidirectional(is_bidirectional),
      _n_intervals(n_intervals) {}

ColumnMap DyadicInterval::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto texts = columns.getArrayColumn<uint32_t>(_input_column);

  ArrayColumnBasePtr<uint32_t> prompts, contexts;

  if (_context_column) {
    contexts = columns.getArrayColumn<uint32_t>(*_context_column);
  }
  size_t chunk_size = (1UL << (_n_intervals - 1)) + 1;

  std::vector<size_t> sample_offsets =
      computeOffsets(texts, contexts, chunk_size);

  std::vector<std::vector<std::vector<uint32_t>>> interval_from_end(
      _n_intervals);
  for (size_t i = 0; i < _n_intervals; i++) {
    interval_from_end.at(i).assign(sample_offsets.back(), {});
  }

  std::vector<std::vector<std::vector<uint32_t>>> interval_from_start;
  if (_is_bidirectional) {
    interval_from_start.resize(_n_intervals);

    for (size_t i = 0; i < _n_intervals; i++) {
      interval_from_start.at(i).assign(sample_offsets.back(), {});
    }
  }
  std::vector<uint32_t> targets(sample_offsets.back());
  std::vector<std::vector<uint32_t>> prompt_inputs;
  if (_prompt_column) {
    prompt_inputs.resize(sample_offsets.back());
    prompts = columns.getArrayColumn<uint32_t>(*_prompt_column);
  }
  std::exception_ptr error;

#pragma omp parallel for default(none)                                    \
    shared(texts, sample_offsets, interval_from_end, interval_from_start, \
           prompts, prompt_inputs, contexts, targets, chunk_size, error)
  for (size_t i = 0; i < texts->numRows(); i++) {
    try {
      auto input_tokens = texts->row(i);

      std::vector<uint32_t> tokens;

      if (_context_column) {
        auto context_tokens = contexts->row(i);
        tokens.insert(tokens.end(), context_tokens.begin(),
                      context_tokens.end());
      }
      size_t target_start = tokens.size();

      tokens.insert(tokens.end(), input_tokens.begin(), input_tokens.end());

      size_t sample_offset = sample_offsets[i];

      for (size_t start = 0; start < tokens.size(); start += chunk_size) {
        size_t end = std::min(start + chunk_size, tokens.size());

        for (size_t target = std::max(start + 1, target_start); target < end;
             target++) {
          for (size_t interval = 0; interval < _n_intervals; interval++) {
            size_t int_len = std::min<size_t>(target - start, 1UL << interval);
            size_t int_start = target - int_len;
            interval_from_end[interval][sample_offset] = {
                tokens.begin() + int_start, tokens.begin() + target};

            if (_is_bidirectional) {
              size_t int_end = start + int_len;
              interval_from_start[interval][sample_offset] = {
                  tokens.begin() + start, tokens.begin() + int_end};
            }
          }

          targets[sample_offset] = tokens[target];
          if (_prompt_column) {
            auto prompt = prompts->row(i);
            prompt_inputs[sample_offset] = {prompt.begin(), prompt.end()};
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
    std::string name =
        _output_interval_prefix + "from_end_" + std::to_string(1 << interval);

    output_columns[name] = ArrayColumn<uint32_t>::make(
        std::move(interval_from_end[interval]), texts->dim());
  }

  if (_is_bidirectional) {
    for (size_t interval = 0; interval < _n_intervals; interval++) {
      std::string name = _output_interval_prefix + "from_start_" +
                         std::to_string(1 << interval);

      output_columns[name] = ArrayColumn<uint32_t>::make(
          std::move(interval_from_start[interval]), texts->dim());
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
    const ArrayColumnBasePtr<uint32_t>& texts,
    const ArrayColumnBasePtr<uint32_t>& contexts, size_t chunk_size) {
  std::vector<size_t> offsets(texts->numRows() + 1);
  offsets[0] = 0;

  for (size_t i = 0; i < texts->numRows(); i++) {
    offsets[i + 1] = 0;

    size_t text_len = texts->row(i).size();

    // In case context tokens passed are greater than 2048, we would ignore all
    // the chunks created with just context.
    if (contexts && (contexts->row(i).size() % (chunk_size) != 0)) {
      // Count offset for the first chunk containing context window
      size_t samples_in_first_chunk = std::min(
          text_len, chunk_size - contexts->row(i).size() % (chunk_size));
      offsets[i + 1] += samples_in_first_chunk;
      text_len -= samples_in_first_chunk;
    }

    // Count n_chunks in remaining text len
    // Eg. context = [10,11,12] ; text = [0,1,2,3,4]
    // 1st chunk = [10,11,12,0,1] -> Offset = 2
    // 2nd chunk = [2,3,4] -> Offset = 2, Total offset = 2+2

    size_t n_chunks = (text_len + chunk_size - 1) / chunk_size;

    // Since we always with at least 1 token as context, hence a chunk of size 4
    // only yields 3 samples.
    offsets[i + 1] += offsets[i] + (text_len - n_chunks);
  }
  return offsets;
}

ColumnMap DyadicInterval::inferenceFeaturization(ColumnMap columns) const {
  auto tokens = columns.getArrayColumn<uint32_t>(_input_column);

  std::vector<std::vector<std::vector<uint32_t>>> intervals_from_end(
      _n_intervals);
  std::vector<std::vector<std::vector<uint32_t>>> intervals_from_start;
  for (size_t i = 0; i < _n_intervals; i++) {
    intervals_from_end.at(i).assign(columns.numRows(), {});
  }
  if (_is_bidirectional) {
    intervals_from_start.resize(_n_intervals);

    for (size_t i = 0; i < _n_intervals; i++) {
      intervals_from_start.at(i).assign(columns.numRows(), {});
    }
  }

#pragma omp parallel for default(none)   \
    shared(tokens, intervals_from_start, \
           intervals_from_end) if (tokens->numRows() > 1)
  for (size_t i = 0; i < tokens->numRows(); i++) {
    auto row_tokens = tokens->row(i);
    auto row_tokens_vector = row_tokens.range(0, row_tokens.size());
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

  for (size_t interval = 0; interval < _n_intervals; interval++) {
    std::string name =
        _output_interval_prefix + "from_end_" + std::to_string(1 << interval);

    columns.setColumn(
        name, ArrayColumn<uint32_t>::make(
                  std::move(intervals_from_end[interval]), tokens->dim()));
  }

  if (_is_bidirectional) {
    for (size_t interval = 0; interval < _n_intervals; interval++) {
      std::string name = _output_interval_prefix + "from_start_" +
                         std::to_string(1 << interval);

      columns.setColumn(
          name, ArrayColumn<uint32_t>::make(
                    std::move(intervals_from_start[interval]), tokens->dim()));
    }
  }

  return columns;
}

template void DyadicInterval::serialize(cereal::BinaryInputArchive&);
template void DyadicInterval::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void DyadicInterval::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column,
          _context_column, _output_interval_prefix, _target_column,
          _n_intervals, _is_bidirectional, _prompt_column);
}

}  // namespace thirdai::data