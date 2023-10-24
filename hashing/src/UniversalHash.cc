#include "UniversalHash.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

namespace thirdai::hashing {

UniversalHash::UniversalHash(uint32_t seed) : _seed(seed) {
  // We can decide to pass in a generator instead, if needed.
  std::mt19937_64 gen(_seed);
  std::uniform_int_distribution<uint64_t> dis;
  for (auto& i : _table) {
    for (auto& j : i) {
      j = dis(gen);
    }
  }
}

UniversalHash::UniversalHash(const proto::hashing::UniversalHash& hash_fn)
    : _seed(hash_fn.seed()) {
  size_t rows = _table.size();
  size_t cols = _table[0].size();

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      _table[i][j] = hash_fn.table().at(i * cols + j);
    }
  }
}

uint64_t UniversalHash::gethash(const std::string& key) const {
  uint64_t res = 0;
  for (uint8_t ch : key) {
    res ^= _table[ch & 7][ch];
  }
  return res;
}

uint64_t UniversalHash::gethash(uint64_t key) const {
  uint64_t res = 0;
  for (uint32_t i = 0; i < sizeof(key); i++) {
    res ^= _table[i][static_cast<unsigned char>(key >> (i << 3))];
  }
  return res;
}

uint32_t UniversalHash::seed() const { return _seed; }

proto::hashing::UniversalHash* UniversalHash::toProto() const {
  auto* hash_fn = new proto::hashing::UniversalHash();

  hash_fn->set_seed(_seed);

  for (const auto& row : _table) {
    for (const auto& item : row) {
      hash_fn->add_table(item);
    }
  }

  return hash_fn;
}

}  // namespace thirdai::hashing
