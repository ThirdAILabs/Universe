#pragma once

#include "../../dataset/Vectors.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace thirdai::utils::lsh_testing {

/** Returns the angle between two sparse vectors */
static float angle(const SparseVector& a, const SparseVector& b) {
  float total = 0, ma = 0, mb = 0;
  uint32_t ia = 0, ib = 0;
  while (ia < a.len && ib < b.len) {
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
  for (uint32_t i = 0; i < a.len; i++) {
    ma += a.values[i] * a.values[i];
  }

  for (uint32_t i = 0; i < b.len; i++) {
    mb += b.values[i] * b.values[i];
  }

  return std::acos(total / (std::sqrt(ma) * std::sqrt(mb)));
}

/** Print out a sparse vector */
static void printVec(const SparseVector& vec) { std::cout << vec << std::endl; }

}  // namespace thirdai::utils::lsh_testing