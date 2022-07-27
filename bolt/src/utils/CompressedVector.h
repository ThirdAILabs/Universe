#pragma once

#include "hashing/src/MurmurHash.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace thirdai::bolt {

static constexpr uint64_t kDefaultBlockSize = 64;
static constexpr uint64_t kDefaultSeed = 42;

// A CompressedVector attempts to compress a large vector into a smaller one by
// means of sketching.
//
// The input vector is partitioned into blocks. The blocks are hashed to
// continuous locations in memory in a compressed vector. Hashing blocks into
// continuous places is expected to improve speed by taking advantage of cache
// line optimizations and predicatable data access patterns.
//
// TODO(jerin): Write-up the math, guarantees.
// TODO(jerin): Remove mod operations with equivalent bit-operations, asserting
//              power of two.
// TODO(jerin): Hash Distribution check. Uniform enough?

template <class ELEMENT_TYPE>
class CompressedVector {
 public:
  // For cereal, but why?
  CompressedVector() {}

  // Create a new CompressedVector.
  CompressedVector(uint64_t physical_size, ELEMENT_TYPE default_value = 0,
                   uint64_t block_size = kDefaultBlockSize,
                   uint32_t seed = kDefaultSeed, bool use_sign_bit = true)
      : _physical_vector(physical_size + block_size, default_value),
        _block_size(block_size),
        _seed(seed),
        _use_sign_bit(use_sign_bit),
        _truncated_size(physical_size) {}

  // Create a new CompressedVector from a pre-existing vector.
  CompressedVector(const std::vector<ELEMENT_TYPE>& input,
                   uint64_t physical_size,
                   uint64_t block_size = kDefaultBlockSize,
                   uint32_t seed = kDefaultSeed, bool use_sign_bit = true)
      : CompressedVector(physical_size, /*default_value=*/0, block_size, seed,
                         use_sign_bit) {
    // Do we have BOLT_ASSERT yet?
    assert(physical_size <= input.size());
    assert(physical_size > block_size);

    for (uint64_t i = 0; i < input.size(); i += _block_size) {
      // Find the location the first element of the block hashes into.
      // Hashing is truncated by truncated_size to avoid out of bounds access in
      // the nested loop below.

      uint64_t block_begin = hashFunction(i) % _truncated_size;

      // Having found the hash, we store all elements in the block within the
      // respective offset.
      for (uint64_t j = i; j < i + _block_size; j++) {
        uint64_t offset = j - i;
        uint64_t index = block_begin + offset;

        if (not _use_sign_bit) {
          _physical_vector[index] += input[j];
        } else {
          bool sign_bit = hashFunction(j) % 2;

          // Add the input value multiplied by sign bit to the index at
          // _physical_vector.
          if (sign_bit) {
            _physical_vector[index] += input[j];
          } else {
            _physical_vector[index] -= input[j];
          }
        }

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

  ELEMENT_TYPE operator[](uint64_t index) { return get(index); }

  // non-const accessor.
  ELEMENT_TYPE get(uint64_t i) const {
    uint64_t idx = findIndexInPhysicalVector(i);
    ELEMENT_TYPE value = _physical_vector[idx];

    if (_use_sign_bit) {
      uint64_t sign_bit = hashFunction(i) % 2;
      value = sign_bit ? value : -1 * value;
    }

    assert(not std::isnan(value));

    return value;
  }

  // Set a value at an index.
  void set(uint64_t i, ELEMENT_TYPE value) {
    uint64_t idx = findIndexInPhysicalVector(i);
    ELEMENT_TYPE& current_value = _physical_vector[idx];

    if (_use_sign_bit) {
      uint64_t sign_bit = hashFunction(i) % 2;
      current_value += sign_bit ? value : -1 * value;
    } else {
      current_value += value;
    }
  }

  void assign(uint64_t size, ELEMENT_TYPE value) {
    (void)size;
    std::fill(_physical_vector.data(),
              _physical_vector.data() + _physical_vector.size(), value);
  }

  void clear() { _physical_vector.clear(); }

  // Iterators for pseudo-view on the bigger vector.

 private:
  std::vector<ELEMENT_TYPE> _physical_vector;  // Underlying vector which stores
                                               // the compressed elements.
  uint64_t _block_size;  // Blocks of elements to use in compressed hashing
                         // for cache friendliness.
  uint32_t _seed;        // For consistency *and* pseudorandomness.

  bool _use_sign_bit;  // Whether to use a sign-bit in hashing to get an
                       // unbiased estimator.

  uint64_t _truncated_size;  // For purposes of hashing.

  // Convenience function to hash into a uint32_t using MurmurHash using saved
  // seed value.
  inline uint32_t hashFunction(uint64_t value) const {
    char* addr = reinterpret_cast<char*>(&value);
    uint32_t hash_value =
        thirdai::hashing::MurmurHash(addr, sizeof(uint64_t), _seed);
    return hash_value;
  }

  uint64_t findIndexInPhysicalVector(uint64_t i) const {
    // The following involves the mod operation and is slow.
    // We will have to do bit arithmetic somewhere.
    // TODO(jerin): Come back here and make more efficient.
    uint64_t offset = i % _block_size;
    uint64_t i_begin = i - offset;

    uint64_t block_begin = hashFunction(i_begin) % _truncated_size;
    uint64_t index = block_begin + offset;
    return index;
  }
};

}  // namespace thirdai::bolt
