#pragma once

#include <dataset/src/Vectors.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace thirdai::hashing {

/** Returns the angle between two sparse vectors */
static float angle(const thirdai::dataset::SparseVector& a,
                   const thirdai::dataset::SparseVector& b) {
  float total = 0, ma = 0, mb = 0;
  uint32_t ia = 0, ib = 0;
  while (ia < a.length() && ib < b.length()) {
    if (a._indices[ia] == b._indices[ib]) {
      total += a._values[ia] * b._values[ib];
      ia++;
      ib++;
    } else if (a._indices[ia] < b._indices[ib]) {
      ia++;
    } else {
      ib++;
    }
  }
  for (uint32_t i = 0; i < a.length(); i++) {
    ma += a._values[i] * a._values[i];
  }

  for (uint32_t i = 0; i < b.length(); i++) {
    mb += b._values[i] * b._values[i];
  }

  return std::acos(total / (std::sqrt(ma) * std::sqrt(mb)));
}

/** Print out a sparse vector */
static void printVec(const thirdai::dataset::SparseVector& vec) {
  std::cout << vec << std::endl;
}

}  // namespace thirdai::hashing