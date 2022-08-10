#pragma once

#include "hashing/src/MurmurHash.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace thirdai::bolt {

// A CompressedVector attempts to compress a large vector into a smaller one by
// means of sketching.
template <class ELEMENT_TYPE>
class CompressedVector {
 public:
  virtual ELEMENT_TYPE operator[](uint64_t index) = 0;

  // non-const accessor.
  virtual ELEMENT_TYPE get(uint64_t i) const = 0;

  // Set a value at an index.
  virtual void set(uint64_t i, ELEMENT_TYPE value) = 0;

  // For compatibility with std::vector usage in source.
  virtual void assign(uint64_t size, ELEMENT_TYPE value) = 0;

  virtual void clear() = 0;

  virtual ~CompressedVector() = default;
};

static constexpr uint64_t kDefaultBlockSize = 1;
static constexpr uint64_t kDefaultSeed = 42;

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
class BiasedSketch final : public CompressedVector<ELEMENT_TYPE> {
 public:
  // For cereal, but why?
  BiasedSketch() {}

  // Create a new BiasedSketch.
  explicit BiasedSketch(uint64_t physical_size, ELEMENT_TYPE default_value = 0,
                        uint64_t block_size = kDefaultBlockSize,
                        uint32_t seed = kDefaultSeed);

  // Create a new BiasedSketch from a pre-existing vector.
  BiasedSketch(const std::vector<ELEMENT_TYPE>& input, uint64_t physical_size,
               uint64_t block_size = kDefaultBlockSize,
               uint32_t seed = kDefaultSeed);

  ELEMENT_TYPE operator[](uint64_t index) final { return get(index); }

  // non-const accessor.
  ELEMENT_TYPE get(uint64_t i) const final;

  // Set a value at an index.
  void set(uint64_t i, ELEMENT_TYPE value) final;

  void assign(uint64_t size, ELEMENT_TYPE value) final;

  void clear() final;

 private:
  std::vector<ELEMENT_TYPE> _physical_vector;  // Underlying vector which stores
                                               // the compressed elements.
  uint64_t _block_size;                        // Blocks of elements to use in
                                               // compressed hashing for cache
                                               // friendliness.
  uint32_t _seed;  // For consistency *and* pseudorandomness.

  uint64_t _truncated_size;  // For purposes of hashing.

  uint64_t findIndexInPhysicalVector(uint64_t i) const;
};

template <class ELEMENT_TYPE>
class UnbiasedSketch final : public CompressedVector<ELEMENT_TYPE> {
 public:
  // For cereal, but why?
  UnbiasedSketch() {}

  // Create a new UnbiasedSketch.
  explicit UnbiasedSketch(uint64_t physical_size,
                          ELEMENT_TYPE default_value = 0,
                          uint64_t block_size = kDefaultBlockSize,
                          uint32_t seed = kDefaultSeed)
      : _physical_vector(physical_size + block_size, default_value),
        _block_size(block_size),
        _seed(seed),
        _truncated_size(physical_size) {}

  // Create a new UnbiasedSketch from a pre-existing vector.
  UnbiasedSketch(const std::vector<ELEMENT_TYPE>& input, uint64_t physical_size,
                 uint64_t block_size = kDefaultBlockSize,
                 uint32_t seed = kDefaultSeed);

  ELEMENT_TYPE operator[](uint64_t index) final { return get(index); }

  // non-const accessor.
  ELEMENT_TYPE get(uint64_t i) const final;

  // Set a value at an index.
  void set(uint64_t i, ELEMENT_TYPE value) final;

  void assign(uint64_t size, ELEMENT_TYPE value) final;

  void clear() final;

 private:
  std::vector<ELEMENT_TYPE> _physical_vector;  // Underlying vector which stores
                                               // the compressed elements.
  uint64_t _block_size;  // Blocks of elements to use in compressed hashing
                         // for cache friendliness.
  uint32_t _seed;        // For consistency *and* pseudorandomness.

  uint64_t _truncated_size;  // For purposes of hashing.

  uint64_t findIndexInPhysicalVector(uint64_t i) const;
};

}  // namespace thirdai::bolt
