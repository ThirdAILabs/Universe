#pragma once

#include <cereal/access.hpp>
#include <cereal/types/array.hpp>
#include <proto/universal_hash.pb.h>
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

  explicit UniversalHash(const proto::hashing::UniversalHash& hash_fn);

  /**
   * Hash string key.
   */
  uint64_t gethash(const std::string& key) const;

  /**
   * Hash integer key. Allows for smaller int sizes.
   */
  uint64_t gethash(uint64_t key) const;

  /**
   * Returns the seed of the universal hash.
   */
  uint32_t seed() const;

  proto::hashing::UniversalHash* toProto() const;

  // Constructor for cereal.
  UniversalHash() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_seed, _table);
  }

  uint32_t _seed;
  std::array<std::array<uint64_t, 256>, 8> _table;
};

}  // namespace thirdai::hashing
