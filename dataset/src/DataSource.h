#pragma once

#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <fstream>
#include <ios>
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

// This class should remain final because we require for demo licensing that the
// passed in source is a FileDataSource so the user can't trickily add extra
// functionality (and making it not final would allow it to be extended and
// its methods overwritten with tricky behavior).
class FileDataSource final : public DataSource {
 public:
  explicit FileDataSource(const std::string& filename)
      : _file(SafeFileIO::ifstream(
            filename, /* model= */ std::ios_base::in | std::ios_base::binary)),
        _filename(filename) {}

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
  // In the same vein as the above comment about demo licensing, these fields
  // should not be settable after the constructor, since we want _filename to
  // always refer to _file.
  std::ifstream _file;
  std::string _filename;
};

/**
 * In the current data loading framework, the DataSource object produces lines
 * which are then parsed and featurized by the Featurizer object. In a CSV file,
 * a quoted column may contain newline characters. Thus, naively splitting the
 * file into lines by newline characters alone will result in CSV parsing
 * errors.
 *
 * This DataSource class wraps an upstream DataSource object and produces
 * complete CSV rows. Roughly, the nextLine() method repeatedly calls the
 * upstream object's nextLine() method and stores the returned values in a
 * buffer until it sees an unquoted newline character. It then returns the
 * concatenation of the contents of this buffer.
 */
class CsvDataSource final : public DataSource {
 public:
  CsvDataSource(DataSourcePtr source, char delimiter)
      : _source(std::move(source)), _delimiter(delimiter) {}

  std::optional<std::string> nextLine() final;

  std::optional<std::vector<std::string>> nextBatch(
      size_t target_batch_size) final;

  std::string resourceName() const final { return _source->resourceName(); }

  void restart() final {
    _remains = {};
    _source->restart();
  }

  static auto make(DataSourcePtr source, char delimiter) {
    return std::make_shared<CsvDataSource>(std::move(source), delimiter);
  }

 private:
  static bool inQuotes(parsers::CSV::ParserState state) {
    switch (state) {
      case parsers::CSV::ParserState::DelimiterInQuotes:
      case parsers::CSV::ParserState::EscapeInQuotes:
      case parsers::CSV::ParserState::RegularInQuotes:
        return true;
      default:
        return false;
    }
  }

  static bool inQuotesAtEndOfLine(parsers::CSV::StateMachine& state_machine,
                                  const std::string& line);

  DataSourcePtr _source;
  char _delimiter;
  std::optional<std::string> _remains;
};

}  // namespace thirdai::dataset
