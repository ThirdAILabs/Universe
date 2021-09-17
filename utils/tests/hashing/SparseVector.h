#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace thirdai::utils::lsh_testing {

/** Represents a sparse data vector */
struct SparseVector {
  std::vector<uint32_t> indices;
  std::vector<float> values;
  uint32_t num_non_zeros;
};

/** Returns the angle between two sparse vectors */
static float angle(const SparseVector& a, const SparseVector& b) {
  float total = 0, ma = 0, mb = 0;
  uint32_t ia = 0, ib = 0;
  while (ia < a.num_non_zeros && ib < b.num_non_zeros) {
    if (a.indices[ia] == b.indices[ib]) {
      total += a.values[ia] * b.values[ib];
      ia++;
      ib++;
    } else if (a.indices[ia] < b.indices[ib]) {
      ia++;
    } else {
      ib++;
    }
  }
  for (uint32_t i = 0; i < a.num_non_zeros; i++) {
    ma += a.values[i] * a.values[i];
  }

  for (uint32_t i = 0; i < b.num_non_zeros; i++) {
    mb += b.values[i] * b.values[i];
  }

  return std::acos(total / (std::sqrt(ma) * std::sqrt(mb)));
}

/**
 * Returns the cosine similarity between 2 dense vectors, where the cosine
 * similarity is defined as 1 - theta / pi.
 */
static float cosine_sim(const SparseVector& a, const SparseVector& b) {
  return 1.0 - angle(a, b) / M_PI;
}

/** Print out a sparse vector */
static void printVec(const SparseVector& vec) {
  for (uint32_t i = 0; i < vec.num_non_zeros; i++) {
    std::cout << vec.indices.at(i) << ":" << vec.values.at(i) << " ";
  }
  std::cout << std::endl;
}

}  // namespace thirdai::utils::lsh_testing