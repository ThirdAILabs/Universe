#pragma once

#include "hashing/src/MurmurHash.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace thirdai::bolt {

static constexpr uint64_t kDefaultBlockSize = 1;
static constexpr uint64_t kDefaultSeed = 42;

namespace fast {
inline bool isPowerOfTwo(uint64_t value) { return value & (value - 1); }
inline uint64_t modulo(uint64_t x, uint32_t y) { return x & (y - 1); }
}  // namespace fast

class BlockHashUtil {
 public:
  // Empty cereal constructor anti-pattern.
  BlockHashUtil() {}

  BlockHashUtil(uint32_t seed, uint64_t container_size, uint64_t block_size)
      : _seed(seed), _container_size(container_size), _block_size(block_size) {
    assert(fast::isPowerOfTwo(block_size));
    assert(fast::isPowerOfTwo(container_size));
  }

  // Convenience function to hash into a uint32_t using
  // MurmurHash using saved seed value.
  inline uint32_t hash(uint64_t value) const {
    char* addr = reinterpret_cast<char*>(&value);
    uint32_t hash_value =
        thirdai::hashing::MurmurHash(addr, sizeof(uint64_t), _seed);
    return hash_value;
  }

  uint64_t projectedIndex(uint64_t i) const {
    // The following involves the mod operation and is slow.
    // We will have to do bit arithmetic somewhere.
    // TODO(jerin): Come back here and make more efficient.
    uint64_t offset = fast::modulo(i, _block_size);
    uint64_t i_begin = i - offset;

    uint64_t block_begin = fast::modulo(hash(i_begin), _container_size);
    uint64_t index = block_begin + offset;
    return index;
  }

  inline uint64_t block_size() const { return _block_size; }

  inline uint32_t container_size() const { return _container_size; }

 private:
  uint32_t _seed;
  uint64_t _container_size;
  uint64_t _block_size;
};

// A CompressedVector defines and interface for classes intended to compress a
// large vector into a smaller one by sketching. If some the distribution of
// samples that we are trying to compress follows some form of power law  -
// which is the case with gradients, weights, moving average estimates of
// gradients, we will be able to use hashing as a viable means to represent the
// same in a smaller memory footprint, subject to some loss of information.
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

  ELEMENT_TYPE get(uint64_t i) const final;

  // Set a value at an index.
  void set(uint64_t i, ELEMENT_TYPE value) final;

  void assign(uint64_t size, ELEMENT_TYPE value) final;

  void clear() final;

 private:
  std::vector<ELEMENT_TYPE> _sketch;  // Underlying vector which stores
                                      // the compressed elements.
  BlockHashUtil _util;
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
                          uint32_t seed = kDefaultSeed);

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
  std::vector<ELEMENT_TYPE> _sketch;  // Underlying vector which stores
                                      // the compressed elements.
  BlockHashUtil _util;
};

}  // namespace thirdai::bolt
