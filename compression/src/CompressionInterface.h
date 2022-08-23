#pragma once

#include "CompressedVector.h"
#include "CompressionUtils.h"
#include "DefaultCompressedVector.h"
#include "DragonVector.h"
#include <cstdint>
#include <memory>
#include <random>

namespace thirdai::compression {

template <class T>

class CompressionInterface {
 public:
  CompressedVector<T> getCompressedVector(
      const std::vector<T>& values, const std::string& compression_scheme = "",
      float compression_density = 1, int seed_for_hashing = 0) {
    if (compression_scheme == "dragon") {
      return DragonVector<T>(values, compression_density, seed_for_hashing);
    }
    return DefaultCompressedVector<T>(values);
  }

  CompressedVector<T> aggregateCompressedVectors(
      const std::vector<CompressedVector<T>>& vec) {
    (void)vec;
  }
};

}  // namespace thirdai::compression