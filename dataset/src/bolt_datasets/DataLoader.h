#pragma once

#include <dataset/src/utils/SafeFileIO.h>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace thirdai::dataset {

class DataLoader {
 public:
  explicit DataLoader(uint32_t target_batch_size)
      : _target_batch_size(target_batch_size) {}

  virtual std::optional<std::vector<std::string>> nextBatch() = 0;

  virtual std::optional<std::string> getHeader() = 0;

  virtual std::string resourceName() const = 0;

  uint32_t getMaxBatchSize() const { return _target_batch_size; }

  virtual ~DataLoader() = default;

 protected:
  uint32_t _target_batch_size;
};

class SimpleFileDataLoader final : public DataLoader {
 public:
  SimpleFileDataLoader(const std::string& filename, uint32_t target_batch_size)
      : DataLoader(target_batch_size),
        _file(SafeFileIO::ifstream(filename)),
        _filename(filename) {}

  std::optional<std::vector<std::string>> nextBatch() final {
    if (_file.eof()) {
      return std::nullopt;
    }

    std::vector<std::string> lines;
    std::string line;
    while (lines.size() < _target_batch_size && std::getline(_file, line)) {
      lines.push_back(std::move(line));
    }
    if (lines.empty()) {
      return std::nullopt;
    }
    return std::make_optional(std::move(lines));
  }

  std::optional<std::string> getHeader() final {
    std::string line;
    if (std::getline(_file, line)) {
      return line;
    }
    return std::nullopt;
  }

  std::string resourceName() const final { return _filename; }

 private:
  std::ifstream _file;
  std::string _filename;
};

}  // namespace thirdai::dataset
