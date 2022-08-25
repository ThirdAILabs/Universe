#pragma once

#include "CompressionUtils.h"
#include <compression/src/CompressedVector.h>
#include <compression/src/CompressionUtils.h>
#include <compression/src/DefaultCompressedVector.h>
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
  if (compression_scheme == "dragon") {
    return std::make_unique<DragonVector<T>>(values, compression_density,
                                             seed_for_hashing);
  }
  return std::make_unique<DefaultCompressedVector<T>>(values);
}

template <class T>
std::vector<T> decompress(const CompressedVector<T>& vec) {
  return vec.decompressVector();
}

template <class T>
std::unique_ptr<CompressedVector<T>> add(
    const std::vector<std::unique_ptr<CompressedVector<T>>>& vec) {
  if (vec.empty()) {
    throw std::logic_error("Cannot aggregate empty vectors");
  }
  std::string compression_scheme = vec[0]->getCompressionScheme();
  if (compression_scheme == "dragon") {
    return std::make_unique<DragonVector<float>>(
        DragonVector<T>::addVectors(vec));
  }
}

}  // namespace thirdai::compression