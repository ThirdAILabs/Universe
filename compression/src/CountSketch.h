#pragma once

#include "CompressedVector.h"
#include "CompressionUtils.h"
#include <hashing/src/UniversalHash.h>
#include <compression/src/CompressionUtils.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>

using UniversalHash = thirdai::hashing::UniversalHash;

namespace thirdai::compression {
template <class T>
class CountSketch final : public CompressedVector<T> {
 public:
  CountSketch<T>() {}

  // Assumes the number of seeds supplied is equal to num_sketches
  // Or seed_for_hashing_indices==num_sketches

  CountSketch(const std::vector<T>& vector_to_compress,
              float compression_density, uint32_t num_sketches,
              std::vector<uint32_t> seed_for_hashing_indices,
              std::vector<uint32_t> seed_for_sign);

  CountSketch(const T* values_to_compress, uint32_t size,
              float compression_density, uint32_t num_sketches,
              std::vector<uint32_t> seed_for_hashing_indices,
              std::vector<uint32_t> seed_for_sign);

  CountSketch(std::vector<std::vector<T>> count_sketches,
              std::vector<uint32_t> seed_for_hashing_indices,
              std::vector<uint32_t> seed_for_sign, uint32_t _uncompressed_size);

  T get(uint32_t index) const final;

  void set(uint32_t index, T value) final;

  void clear() final;

  void extend(const CountSketch<T>& other_sketch);

  void add(const CountSketch<T>& other_sketch);

  uint32_t numSketches() const;

  uint32_t size() const;

  std::string type() const final;

  std::vector<std::vector<T>> sketches() const { return _count_sketches; }

  std::vector<uint32_t> indexSeeds() const { return _seed_for_hashing_indices; }

  std::vector<uint32_t> signSeeds() const { return _seed_for_sign; }

  uint32_t uncompressedSize() const { return _uncompressed_size; }

  std::vector<T> decompress() const final;

  void printCountsketch() const {
    std::cout << "printing the count sketch" << std::endl;
    for (size_t i = 0; i < _count_sketches.size(); i++) {
      for (size_t j = 0; j < _count_sketches[0].size(); j++) {
        std::cout << _count_sketches[i][j] << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "seed for hashing\n";
    for (size_t i = 0; i < _seed_for_hashing_indices.size(); i++) {
      std::cout << _seed_for_hashing_indices[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "seed for sign\n";
    for (size_t i = 0; i < _seed_for_sign.size(); i++) {
      std::cout << _seed_for_sign[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "uncompressed size: " << _uncompressed_size << std::endl;
  }

  static void printvector(const std::string& name, std::vector<T> vec) {
    std::cout << name << std::endl;
    for (auto i : vec) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }

 private:
  std::vector<std::vector<T>> _count_sketches;
  std::vector<uint32_t> _seed_for_hashing_indices;
  std::vector<uint32_t> _seed_for_sign;
  std::vector<UniversalHash> _hasher_index;
  std::vector<UniversalHash> _hasher_sign;
  uint32_t _uncompressed_size;

  void sketch(const T* values_to_compress, uint32_t size);
};
}  // namespace thirdai::compression