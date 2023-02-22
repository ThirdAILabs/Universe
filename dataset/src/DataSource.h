#pragma once

#include <dataset/src/utils/SafeFileIO.h>
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

}  // namespace thirdai::dataset
