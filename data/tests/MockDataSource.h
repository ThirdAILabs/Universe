#pragma once

#include <dataset/src/DataSource.h>

namespace thirdai::data::tests {

class MockDataSource final : public dataset::DataSource {
 public:
  explicit MockDataSource(std::vector<std::string> lines)
      : _lines(std::move(lines)) {}

  std::string resourceName() const final { return "mock-data-source"; }

  std::optional<std::vector<std::string>> nextBatch(
      size_t target_batch_size) final {
    std::vector<std::string> lines;

    while (lines.size() < target_batch_size) {
      if (auto line = nextLine()) {
        lines.push_back(*line);
      } else {
        break;
      }
    }

    if (lines.empty()) {
      return std::nullopt;
    }

    return lines;
  }

  std::optional<std::string> nextLine() final {
    if (_loc < _lines.size()) {
      return _lines[_loc++];
    }
    return std::nullopt;
  }

  void restart() final { _loc = 0; }

 private:
  std::vector<std::string> _lines;
  size_t _loc = 0;
};

}  // namespace thirdai::data::tests