#include "UniversalHash.h"
#include <_types/_uint64_t.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
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

ar::ConstArchivePtr UniversalHash::toArchive() const {
  auto map = ar::Map ::make();

  map->set("seed", ar::u64(_seed));

  std::vector<uint64_t> flat_table;
  for (const auto& row : _table) {
    flat_table.insert(flat_table.end(), row.begin(), row.end());
  }

  map->set("table", ar::vecU64(std::move(flat_table)));

  return map;
}

UniversalHash::UniversalHash(const ar::Archive& archive)
    : _seed(archive.u64("seed")) {
  size_t rows = _table.size();
  size_t cols = _table[0].size();

  const auto& flat_table = archive.getAs<ar::VecU64>("table");

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      _table[i][j] = flat_table.at(i * cols + j);
    }
  }
}

}  // namespace thirdai::hashing
