#pragma once

#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <utils/StringManipulation.h>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace thirdai::dataset {

class DataSource {
 public:
  virtual std::optional<std::vector<std::string>> nextBatch(
      size_t target_batch_size) = 0;

  virtual std::optional<std::string> nextLine() = 0;

  virtual std::string resourceName() const = 0;

  virtual void restart() = 0;

  virtual ~DataSource() = default;
};

using DataSourcePtr = std::shared_ptr<DataSource>;

class FileDataSource final : public DataSource {
 public:
  explicit FileDataSource(const std::string& filename)
      : _file(SafeFileIO::ifstream(filename)), _filename(filename) {}

  static std::shared_ptr<FileDataSource> make(const std::string& filename) {
    return std::make_shared<FileDataSource>(filename);
  }

  std::optional<std::vector<std::string>> nextBatch(
      size_t target_batch_size) final {
    if (_file.eof()) {
      return std::nullopt;
    }

    std::vector<std::string> lines;
    std::string line;
    while (lines.size() < target_batch_size && std::getline(_file, line)) {
      if (!line.empty()) {
        lines.push_back(std::move(line));
      }
    }

    if (lines.empty()) {
      return std::nullopt;
    }

    return std::make_optional(std::move(lines));
  }

  std::optional<std::string> nextLine() final {
    std::string line;
    if (std::getline(_file, line)) {
      return line;
    }
    return std::nullopt;
  }

  void restart() final {
    _file.clear();
    _file.seekg(0, std::ios::beg);
  }

  std::string resourceName() const final { return _filename; }

 private:
  std::ifstream _file;
  std::string _filename;
};

class CsvDataSource final : public DataSource {
 public:
  CsvDataSource(DataSourcePtr source, char delimiter)
      : _source(std::move(source)), _delimiter(delimiter) {}

  std::optional<std::string> nextLine() final {
    parsers::CSV::StateMachine state_machine(_delimiter);
    std::vector<std::string> buffer;
    std::optional<uint32_t> newline_position;
    while (!newline_position) {
      if (auto line = nextRawLine()) {
        newline_position = findNewline(state_machine, *line);
        if (newline_position && newline_position != line->size()) {
          buffer.push_back(line->substr(0, *newline_position));
          _remains = line->substr(*newline_position);
        } else {
          buffer.push_back(*line);
        }
      } else {
        break;
      }
    }
    if (buffer.empty()) {
      return std::nullopt;
    }
    return text::join(buffer, "\n");
  }

  std::optional<std::vector<std::string>> nextBatch(
      size_t target_batch_size) final {
    std::vector<std::string> lines;
    while (lines.size() < target_batch_size) {
      if (auto next_line = nextLine()) {
        lines.push_back(*next_line);
      } else {
        break;
      }
    }

    if (lines.empty()) {
      return std::nullopt;
    }

    return std::make_optional(std::move(lines));
  }

  std::string resourceName() const final { return _source->resourceName(); }

  void restart() final {
    _remains = {};
    _source->restart();
  }

 private:
  static std::optional<uint32_t> findNewline(
      parsers::CSV::StateMachine& state_machine, const std::string& line) {
    for (uint32_t position = 0; position < line.size(); position++) {
      state_machine.transition(line[position]);
      if (state_machine.state() == parsers::CSV::ParserState::NewLine) {
        return position;
      }
    }
    state_machine.transition('\n');
    if (state_machine.state() == parsers::CSV::ParserState::NewLine) {
      return line.size();
    }
    return std::nullopt;
  }

  std::optional<std::string> nextRawLine() {
    if (_remains) {
      auto line = std::move(_remains);
      _remains = {};
      return line;
    }
    return _source->nextLine();
  }

  DataSourcePtr _source;
  char _delimiter;
  std::optional<std::string> _remains;
};

}  // namespace thirdai::dataset
