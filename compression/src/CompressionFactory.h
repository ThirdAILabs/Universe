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

template <class T>
std::vector<T> decompress(const CompressedVector<T>& compressed_vector) {
  return compressed_vector.decompress();
}

}  // namespace thirdai::compression