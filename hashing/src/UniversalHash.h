#pragma once

#include <cstdint>
#include <string>

namespace thirdai::hashing {

/*
 * Cheaper Hash functions, if you need 64 bit hash use murmurhash or xxhash.
 */
class UniversalHash {
 public:
  explicit UniversalHash(uint32_t seed);

  /**
   * Hash string key.
   */
  uint32_t gethash(const std::string& key);

  /**
   * Hash integer key. Allows for smaller int sizes.
   */
  uint32_t gethash(uint64_t key);

 private:
  uint32_t _seed;
  uint32_t T[8][256];
};

}  // namespace thirdai::hashing
