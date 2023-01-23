#include <dataset/src/DataSource.h>
#include <dataset/src/RecursionWrapper.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <algorithm>
#include <deque>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

std::optional<std::vector<std::string>> RecursionWrapper::nextBatch() {
  if (auto source_batch = _source->nextBatch()) {
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

  auto batch_size = std::min<uint32_t>(_target_batch_size, _leftovers.size());

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
  if (!header) {
    throw std::invalid_argument(
        "The dataset must have a header that contains column names.");
  }

  auto column_names = ProcessorUtils::parseCsvRow(*header, _column_delimiter);
  for (uint32_t i = 0; i < column_names.size(); i++) {
    if (column_names[i] == _sequence_column_name) {
      _sequence_column_number = i;
    }
  }

  std::stringstream augmented_header;
  augmented_header << *header;
  for (const auto& column_name : _intermediate_column_names) {
    augmented_header << _column_delimiter << column_name;
  }

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
  auto sequence_column = std::string(columns[_sequence_column_number]);
  auto sequence =
      ProcessorUtils::parseCsvRow(sequence_column, _sequence_delimiter);
  std::vector<std::string> augmented(sequence.size());
  for (uint32_t i = 0; i < sequence.size(); i++) {
    std::stringstream augmented_line;
    copyExceptSequenceColumn(augmented_line, columns, sequence[i]);
    for (uint32_t r = 0; r < _intermediate_column_names.size(); r++) {
      augmented_line << _column_delimiter;
      if (r < i) {
        augmented_line << sequence[r];
      }
    }
    augmented[i] = augmented_line.str();
  }
  return augmented;
}

void RecursionWrapper::copyExceptSequenceColumn(
    std::stringstream& stream, std::vector<std::string_view>& columns,
    std::string_view sequence_column) const {
  columns[_sequence_column_number] = sequence_column;
  if (columns.empty()) {
    return;
  }
  stream << columns.front();
  for (uint32_t i = 1; i < columns.size(); i++) {
    stream << _column_delimiter << columns[i];
  }
}

}  // namespace thirdai::dataset