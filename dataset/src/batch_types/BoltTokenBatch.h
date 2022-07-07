#pragma once

#include <vector>

namespace thirdai::dataset {

class BoltTokenBatch {
 public:
  explicit BoltTokenBatch(std::vector<std::vector<uint32_t>>&& tokens)
      : _tokens(std::move(tokens)) {}

  std::vector<uint32_t>& operator[](size_t i) {
    assert(i < _tokens.size());
    return _tokens[i];
  }

  const std::vector<uint32_t>& operator[](size_t i) const {
    assert(i < _tokens.size());
    return _tokens[i];
  }

  uint32_t getBatchSize() const { return _tokens.size(); }

  BoltTokenBatch(const BoltTokenBatch& other) = delete;

  BoltTokenBatch(BoltTokenBatch&& other) = default;

  BoltTokenBatch& operator=(const BoltTokenBatch& other) = delete;

  BoltTokenBatch& operator=(BoltTokenBatch&& other) = default;

 private:
  std::vector<std::vector<uint32_t>> _tokens;
};

}  // namespace thirdai::dataset