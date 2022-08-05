#pragma once

#include <cereal/access.hpp>
#include <cereal/types/array.hpp>
#include <array>
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
  // Private constructor for cereal.
  UniversalHash() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_seed, T);
  }

  uint32_t _seed;
  std::array<std::array<uint32_t, 256>, 8> T;
};

}  // namespace thirdai::hashing
