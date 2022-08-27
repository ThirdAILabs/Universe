#pragma once

#include "CompressionUtils.h"
#include <compression/src/CompressedVector.h>
#include <compression/src/CompressionUtils.h>
#include <compression/src/DragonVector.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>

namespace thirdai::compression {

template <class T>
std::unique_ptr<CompressedVector<T>> compress(
    const std::vector<T>& values, const std::string& compression_scheme = "",
    float compression_density = 1, int seed_for_hashing = 0) {
  return compress(values.data(), static_cast<uint32_t>(values.size()),
                  compression_scheme, compression_density, seed_for_hashing);
}

template <class T>
std::unique_ptr<CompressedVector<T>> compress(
    const T* values, uint32_t size, const std::string& compression_scheme = "",
    float compression_density = 1, int seed_for_hashing = 0) {
  if (compression_scheme == "dragon") {
    return std::make_unique<DragonVector<T>>(values, size, compression_density,
                                             seed_for_hashing);
  }
  throw std::logic_error("Compression Scheme is invalid");
}

std::unique_ptr<CompressedVector> add(
    const std::vector<std::unique_ptr<Vector>>& compressed_vectors) {}

template <class Vector>
Vector add(const std::vector<std::unique_ptr<Vector>>& compressed_vectors) {
  Vector result = *(compressed_vectors[0].get());
  for (size_t i = 0; i < compressed_vectors.size(); i++) {
    result = result.add(compressed_vectors[i]);
  }
  return result;
}

template <class Vector>
Vector extend(const std::vector<std::unique_ptr<Vector>>& compressed_vectors) {
  Vector result = *(compressed_vectors[0].get());
  for (size_t i = 0; i < compressed_vectors.size(); i++) {
    result = result.extend(compressed_vectors[i]);
  }
  return result;
}

template <class T>
std::vector<T> decompress(const CompressedVector<T>& compressed_vector) {
  return compressed_vector.decompress();
}

}  // namespace thirdai::compression