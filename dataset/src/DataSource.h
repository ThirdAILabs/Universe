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
  explicit DataSource(uint32_t target_batch_size)
      : _target_batch_size(target_batch_size) {}

  virtual std::optional<std::vector<std::string>> nextBatch() = 0;

  virtual std::optional<std::string> nextLine() = 0;

  virtual std::string resourceName() const = 0;

  virtual void restart() = 0;

  uint32_t getMaxBatchSize() const { return _target_batch_size; }

  virtual ~DataSource() = default;

 protected:
  uint32_t _target_batch_size;
};

using DataSourcePtr = std::shared_ptr<DataSource>;

class SimpleFileDataSource final : public DataSource {
 public:
  SimpleFileDataSource(const std::string& filename, uint32_t target_batch_size)
      : DataSource(target_batch_size),
        _file(SafeFileIO::ifstream(filename)),
        _filename(filename) {}

  static std::shared_ptr<SimpleFileDataSource> make(
      const std::string& filename, uint32_t target_batch_size) {
    return std::make_shared<SimpleFileDataSource>(filename, target_batch_size);
  }

  std::optional<std::vector<std::string>> nextBatch() final {
    if (_file.eof()) {
      return std::nullopt;
    }

    std::vector<std::string> lines;
    std::string line;
    while (lines.size() < _target_batch_size && std::getline(_file, line)) {
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
