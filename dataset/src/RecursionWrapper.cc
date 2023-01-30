#include <dataset/src/DataSource.h>
#include <dataset/src/RecursionWrapper.h>
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <algorithm>
#include <deque>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

std::optional<std::vector<std::string>> RecursionWrapper::nextBatch(
    size_t target_batch_size) {
  if (auto source_batch = _source->nextBatch(target_batch_size)) {
    std::vector<std::vector<std::string>> augmented_lines(source_batch->size());
#pragma omp parallel for default(none) shared(source_batch, augmented_lines)
    for (uint32_t i = 0; i < source_batch->size(); i++) {
      augmented_lines[i] = augment(source_batch->at(i));
    }
    for (auto& augmentations : augmented_lines) {
      for (auto& line : augmentations) {
        _leftovers.push_back(std::move(line));
      }
    }
  }

  if (_leftovers.empty()) {
    return std::nullopt;
  }

  auto batch_size = std::min<uint32_t>(target_batch_size, _leftovers.size());

  std::vector<std::string> augmented_batch(
      std::make_move_iterator(_leftovers.begin()),
      std::make_move_iterator(_leftovers.begin() + batch_size));
  for (uint32_t i = 0; i < batch_size; i++) {
    _leftovers.pop_front();
  }

  return augmented_batch;
}

std::optional<std::string> RecursionWrapper::nextLine() {
  if (_at_beginning) {
    return header();
  }
  return nextLineBody();
}

std::string RecursionWrapper::header() {
  auto header = _source->nextLine();
  if (!header || header->empty()) {
    throw std::invalid_argument(
        "The dataset must have a header that contains column names.");
  }

  auto column_names = ProcessorUtils::parseCsvRow(*header, _column_delimiter);
  for (uint32_t i = 0; i < column_names.size(); i++) {
    if (column_names[i] == _target_column) {
      _target_column_number = i;
    }
  }

  std::stringstream augmented_header;
  augmented_header << substringLeftOfTarget(column_names)
                   // Insert intermediate sequence column before target column.
                   // Target column is followed by step column.
                   << _intermediate_column << _column_delimiter
                   << _target_column << _column_delimiter << _step_column
                   << substringRightOfTarget(column_names);

  _at_beginning = false;
  return augmented_header.str();
}

std::optional<std::string> RecursionWrapper::nextLineBody() {
  if (auto next_line_from_source = _source->nextLine()) {
    auto augmented_lines = augment(*next_line_from_source);
    _leftovers.insert(_leftovers.end(),
                      std::make_move_iterator(augmented_lines.begin()),
                      std::make_move_iterator(augmented_lines.end()));
  }

  if (_leftovers.empty()) {
    return std::nullopt;
  }

  auto next_line = std::move(_leftovers.front());
  _leftovers.pop_front();
  return next_line;
}

std::vector<std::string> RecursionWrapper::augment(
    const std::string& original) const {
  auto columns = ProcessorUtils::parseCsvRow(original, _column_delimiter);

  std::string left_of_target = substringLeftOfTarget(columns);
  std::string right_of_target = substringRightOfTarget(columns);

  auto target_seq = std::string(columns[_target_column_number]);
  auto sequence = ProcessorUtils::parseCsvRow(target_seq, _target_delimiter);
  if (sequence.size() < _max_recursion_depth) {
    sequence.push_back(EARLY_STOP);
  }
  if (sequence.size() > _max_recursion_depth) {
    std::cout << "WARNING: found target sequence \"" << target_seq << "\" with "
              << sequence.size() << " elements. Expected sequence length = "
              << _max_recursion_depth << ". Ignoring extra elements."
              << std::endl;
    sequence.resize(_max_recursion_depth);
  }

  std::vector<std::string> augmentations(sequence.size());

  for (uint32_t step = 0; step < sequence.size(); step++) {
    auto& target_item = sequence[step];

    /*
      Intermediate column contains items in `sequence` before
      `target_item`, delimited by `_target_delimiter`.
      E.g.
      `sequence` = ["a", "b", "c", "d"], `_terget_delimiter` = ' ', `step` = 2
      Therefore, `target_item` = "c", intermediate column = "a b"
    */
    std::stringstream intermediate_column;
    for (uint32_t seq_idx = 0; seq_idx < step; seq_idx++) {
      if (seq_idx > 0) {
        intermediate_column << _target_delimiter;
      }
      intermediate_column << sequence[seq_idx];
    }

    // Combine into a single line.
    std::stringstream augmented_line;
    augmented_line << left_of_target << intermediate_column.str()
                   << _column_delimiter << target_item << _column_delimiter
                   << step << right_of_target;

    augmentations[step] = augmented_line.str();
  }
  return augmentations;
}

std::string RecursionWrapper::substringLeftOfTarget(
    const std::vector<std::string_view>& columns) const {
  std::stringstream before_target;
  for (uint32_t i = 0; i < _target_column_number; i++) {
    before_target << columns[i] << _column_delimiter;
  }
  return before_target.str();
}

std::string RecursionWrapper::substringRightOfTarget(
    const std::vector<std::string_view>& columns) const {
  std::stringstream after_target;
  for (uint32_t i = _target_column_number + 1; i < columns.size(); i++) {
    after_target << _column_delimiter << columns[i];
  }
  return after_target.str();
}

}  // namespace thirdai::dataset