#pragma once

#include "CompressedVector.h"
#include "CompressionUtils.h"
#include <hashing/src/UniversalHash.h>
#include <compression/src/CompressionUtils.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>

using UniversalHash = thirdai::hashing::UniversalHash;

namespace thirdai::compression {

/*
 * Count-sketch algorithm:
 * https://stackoverflow.com/questions/6811351/explaining-the-count-sketch-algorithm
 * https://en.wikipedia.org/wiki/Count_sketch
 *
 * NOTE: We are using mean instead of the median to get an estimator for the
 * element.
 * TODO: Run accuracy and performance benchmarks for Count-median sketch vs
 * Count-mean sketch
 */
template <class T>
class CountSketch final : public CompressedVector<T> {
 public:
  CountSketch() {}
  // Assumes the number of seeds supplied is equal to num_sketches
  // Or seed_for_hashing_indices==num_sketches
  CountSketch(const std::vector<T>& vector_to_compress,
              float compression_density, uint32_t num_sketches,
              const std::vector<uint32_t>& seed_for_hashing_indices,
              const std::vector<uint32_t>& seed_for_sign);

  CountSketch(const T* values_to_compress, uint32_t size,
              float compression_density, uint32_t num_sketches,
              const std::vector<uint32_t>& seed_for_hashing_indices,
              const std::vector<uint32_t>& seed_for_sign);

  explicit CountSketch(const char* serialized_data);

  T get(uint32_t index) const final;

  void set(uint32_t index, T value) final;

  void clear() final;

  void extend(const CountSketch<T>& other_sketch);

  void add(const CountSketch<T>& other_sketch);

  void divide(uint32_t divisor);

  uint32_t numSketches() const;

  uint32_t size() const final;

  CompressionScheme type() const final;

  std::vector<std::vector<T>> sketches() const { return _count_sketches; }

  uint32_t uncompressedSize() const final { return _uncompressed_size; }

  std::vector<T> decompress() const final;

  void serialize(char* serialized_data) const final;

  uint32_t serialized_size() const final;

 private:
  std::vector<std::vector<T>> _count_sketches;
  std::vector<UniversalHash> _hasher_index;
  std::vector<UniversalHash> _hasher_sign;
  uint32_t _uncompressed_size;

  void sketch(const T* values_to_compress, uint32_t size);

  int hash_sign(uint32_t sketch_id, uint32_t index) const;

  uint32_t hash_index(uint32_t sketch_id, uint32_t index) const;
};
}  // namespace thirdai::compression