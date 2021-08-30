#include "UniversalHash.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

namespace ThirdAI {

UniversalHash::UniversalHash(uint32_t seed) {
  _seed = seed;
  srand(seed);
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<unsigned long> dis;
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 256; j++) T[i][j] = dis(gen);
}

// uint32_t UniversalHash::gethash(uint8_t key){
// 	return T[key & 7][key];
// }

uint32_t UniversalHash::gethash(std::string key) {
  // Not really a great hash, but will be faster.
  uint32_t res = 0;
  for (size_t i = 0; i < key.length(); i++) {
    char temp = key.at(i);
    res ^= T[temp & 7][(uint32_t)temp];
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
  res ^= T[0][(char)(key)];
  res ^= T[1][(char)(key >> 8)];
  res ^= T[2][(char)(key >> 16)];
  res ^= T[3][(char)(key >> 24)];
  res ^= T[4][(char)(key >> 32)];
  res ^= T[5][(char)(key >> 40)];
  res ^= T[6][(char)(key >> 48)];
  res ^= T[7][(char)(key >> 56)];
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

}  // namespace ThirdAI
