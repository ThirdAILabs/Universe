#pragma once

#include <dataset/src/DataSource.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <algorithm>
#include <deque>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

class RecursiveDataSource final : public DataSource {
 public:
  RecursiveDataSource(DataSourcePtr source, char column_delimiter,
                      char sequence_delimiter, std::string sequence_column_name,
                      std::vector<std::string> recursive_column_names,
                      uint32_t target_batch_size)
      : DataSource(target_batch_size),
        _column_delimiter(column_delimiter),
        _sequence_delimiter(sequence_delimiter),
        _sequence_column_name(std::move(sequence_column_name)),
        _recursive_column_names(std::move(recursive_column_names)),
        _source(std::move(source)),
        _at_beginning(true) {}

  static auto make(DataSourcePtr source, char column_delimiter,
                   char sequence_delimiter, std::string sequence_column_name,
                   std::vector<std::string> recursive_column_names,
                   uint32_t target_batch_size) {
    return std::make_shared<RecursiveDataSource>(
        std::move(source), column_delimiter, sequence_delimiter,
        std::move(sequence_column_name), recursive_column_names,
        target_batch_size);
  }

  std::string resourceName() const final { return _source->resourceName(); }

  void restart() final {
    _leftovers.clear();
    _source->restart();
  }

  std::optional<std::vector<std::string>> nextBatch() final {
    if (auto source_batch = _source->nextBatch()) {
      std::vector<std::vector<std::string>> augmented_lines(
          source_batch->size());
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

    for (const auto& line : augmented_batch) {
      std::cout << line << std::endl;
    }

    return augmented_batch;
  }

  std::optional<std::string> nextLine() final {
    if (!_at_beginning) {
      if (auto next_line_from_source = _source->nextLine()) {
        auto augmented_lines = augment(*next_line_from_source);
        _leftovers.insert(_leftovers.end(),
                          std::make_move_iterator(augmented_lines.begin()),
                          std::make_move_iterator(augmented_lines.end()));
      }
      auto next_line = std::move(_leftovers.front());
      _leftovers.pop_front();
      return next_line;
    }

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
    _at_beginning = false;
    std::stringstream augmented_header;
    augmented_header << *header;
    for (const auto& column_name : _recursive_column_names) {
      augmented_header << _column_delimiter << column_name;
    }
    std::cout << augmented_header.str() << std::endl;
    return augmented_header.str();
  }

 private:
  std::vector<std::string> augment(const std::string& original) const {
    auto columns = ProcessorUtils::parseCsvRow(original, _column_delimiter);
    auto sequence_column = std::string(columns[_sequence_column_number]);
    auto sequence =
        ProcessorUtils::parseCsvRow(sequence_column, _sequence_delimiter);
    std::vector<std::string> augmented(sequence.size());
    for (uint32_t i = 0; i < sequence.size(); i++) {
      std::stringstream augmented_line;
      copyExceptSequenceColumn(augmented_line, columns, sequence[i]);
      for (uint32_t r = 0; r < _recursive_column_names.size(); r++) {
        augmented_line << _column_delimiter;
        if (r < i) {
          augmented_line << sequence[r];
        }
      }
      augmented[i] = augmented_line.str();
    }
    return augmented;
  }

  void copyExceptSequenceColumn(std::stringstream& stream,
                                std::vector<std::string_view>& columns,
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

  char _column_delimiter;
  char _sequence_delimiter;
  std::string _sequence_column_name;
  std::vector<std::string> _recursive_column_names;

  DataSourcePtr _source;
  std::deque<std::string> _leftovers;
  bool _at_beginning;

  uint32_t _sequence_column_number;
};

}  // namespace thirdai::dataset