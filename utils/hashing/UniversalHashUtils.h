#pragma once

#include <cstring>

namespace thirdai::utils {
/*
 * Cheaper Hash functions, if you need 64 bit hash use murmurhash or xxhash
 */
class UniversalHash {
 private:
  uint32_t _seed;
  uint32_t T[8][256];

 public:
  explicit UniversalHash(uint32_t seed);

  /**
   * Hash string key.
   */
  static uint32_t gethash(const std::string& key);

  /**
   * Hash integer key. Allows for smaller int sizes.
   */
  static uint32_t gethash(uint64_t key);

  ~UniversalHash();
};

}  // namespace thirdai::utils
