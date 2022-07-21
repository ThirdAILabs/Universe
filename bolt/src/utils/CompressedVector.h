#pragma once

#include "hashing/src/MurmurHash.h"
#include <cassert>
#include <cstddef>
#include <vector>

namespace thirdai::bolt {

// A CompressedVector attempts to compress a large vector into a smaller one by
// means of sketching.
//
// The input vector is partitioned into blocks. The blocks are hashed to
// continuous locations in memory in a compressed vector.
//
// TODO(jerin): Write-up the math, guarantees.
template <class ELEMENT_TYPE>
class CompressedVector {
 public:
  // Create a new CompressedVector.
  CompressedVector(uint64_t physical_size, uint64_t block_size, uint32_t seed)
      : _physical_vector(physical_size, 0),
        _block_size(block_size),
        _seed(seed) {}

  // Create a new CompressedVector from a pre-existing vector.
  CompressedVector(const std::vector<ELEMENT_TYPE>& input,
                   uint64_t physical_size, uint64_t block_size, uint32_t seed)
      : _physical_vector(physical_size, 0),
        _block_size(block_size),
        _seed(seed) {
    // Do we have BOLT_ASSERT yet?
    assert(physical_size < input.size());
    assert(physical_size > block_size);

    for (uint64_t i = 0; i < input.size(); i += _block_size) {
      // Find the location the first element of the block hashes into.
      // effective_size is required as we are hashing blocks and we don't want
      // out of bounds access.
      uint64_t effective_size = _physical_vector.size() - _block_size;
      uint64_t block_begin = _hash_function(i) % effective_size;

      // Having found the hash, we store all elements in the block within the
      // respective offset.
      for (uint64_t j = i; j < i + _block_size; j++) {
        uint64_t offset = j - i;
        uint64_t index = block_begin + offset;

        // Add the input value to the hash-location.
        _physical_vector[index] += input[j];

        // TODO(jerin): What happens if overflow? We are using sum to store
        // multiple elements, which could overflow the element's type.
      }
    }
  }

  // Add a non-compressed vector to this CompressedVector.
  CompressedVector operator+(const std::vector<ELEMENT_TYPE>& input) const;
  CompressedVector& operator+=(const std::vector<ELEMENT_TYPE>& input);

  // Add two compressed vectors.
  CompressedVector operator+(const CompressedVector& input) const;
  CompressedVector& operator+=(const CompressedVector& input);

  // non-const accessor.
  ELEMENT_TYPE& operator[](uint64_t i) {
    uint64_t idx = _find_index_in_physical_vector(i);
    return _physical_vector[idx];
  }

  // Const accessor to an element.
  const ELEMENT_TYPE& operator[](uint64_t i) const {
    uint64_t idx = _find_index_in_physical_vector(i);
    return _physical_vector[idx];
  }

  // Iterators for pseudo-view on the bigger vector.

 private:
  std::vector<ELEMENT_TYPE> _physical_vector;  // Underlying vector which stores
                                               // the compressed elements.
  uint64_t _block_size;  // Blocks of elements to use in compressed hashing for
                         // cache friendliness.
  uint32_t _seed;        // For consistency *and* pseudorandomness.

  // Convenience function to hash into a uint64_t using MurmurHash.
  // Might be worthwhile to skip this function if only used in one place.
  inline uint32_t _hash_function(uint64_t value) const {
    char* addr = reinterpret_cast<char*>(&value);
    uint32_t hash_value =
        thirdai::hashing::MurmurHash(addr, sizeof(uint64_t), _seed);
    return hash_value;
  }

  uint64_t _find_index_in_physical_vector(uint64_t i) const {
    // The following involves the mod operation and is slow.
    // We will have to do bit arithmetic somewhere.
    // TODO(jerin): Come back here and make more efficient.
    uint64_t offset = i % _block_size;
    uint64_t i_begin = i - offset;

    uint64_t effective_size = _physical_vector.size() - _block_size;
    uint64_t block_begin = _hash_function(i_begin) % effective_size;
    uint64_t index = block_begin + offset;
    return index;
    ;
  }
};

}  // namespace thirdai::bolt
