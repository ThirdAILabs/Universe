#include "UniversalHash.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

namespace thirdai::examples {

UniversalHash::UniversalHash(uint32_t seed) {
  _seed = seed;
  srand(seed);
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 256; j++) {
      T[i][j] = dis(gen);
    }
  }
}

// uint32_t UniversalHash::gethash(uint8_t key){
// 	return T[key & 7][key];
// }

uint32_t UniversalHash::gethash(std::string key) {
  // Not really a great hash, but will be faster.
  uint32_t res = 0;
  for (size_t i = 0; i < key.length(); i++) {
    uint8_t temp = key.at(i);
    res ^= T[temp & 7][static_cast<uint32_t>(temp)];
  }
  return res;
}

// uint32_t UniversalHash::gethash(uint32_t key){
// 	uint32_t res = 0;
// 	res ^= T[0][(char)(key)];
// 	res ^= T[1][(char)(key >> 8)];
// 	res ^= T[2][(char)(key >> 16)];
// 	res ^= T[3][(char)(key >> 24)];
// 	return res;
// }

uint32_t UniversalHash::gethash(uint64_t key) {
  uint32_t res = 0;
  for (uint32_t i = 0; i < 8; i++) {
    res ^= T[i][static_cast<char>(key >> (i << 3))];
  }
  return res;
}

void UniversalHash::getBatchHash(const uint8_t* keys, const uint32_t* hashes,
                                 uint32_t batchSize, uint32_t numHashes) {}

void UniversalHash::getBatchHash(const uint32_t* keys, const uint32_t* hashes,
                                 uint32_t batchSize, uint32_t numHashes) {}

void UniversalHash::getBatchHash(const uint64_t* keys, const uint32_t* hashes,
                                 uint32_t batchSize, uint32_t numHashes) {}

void UniversalHash::getBatchHash(const char* keys, const uint32_t* hashes,
                                 uint32_t batchSize, uint32_t numHashes) {}

}  // namespace thirdai::examples
