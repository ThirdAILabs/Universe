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
  for (auto& i : T) {
    for (auto& j : i) {
      j = dis(gen);
    }
  }
}

uint32_t UniversalHash::gethash(const std::string& key) {
  uint32_t res = 0;
  for (uint8_t ch : key) {
    res ^= T[ch & 7][ch];
  }
  return res;
}

uint32_t UniversalHash::gethash(uint64_t key) {
  uint32_t res = 0;
  for (uint32_t i = 0; i < sizeof(key); i++) {
    res ^= T[i][static_cast<unsigned char>(key >> (i << 3))];
  }
  return res;
}

}  // namespace thirdai::hashing
