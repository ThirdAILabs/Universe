#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace thirdai::hashing {

struct SparseVector {
  std::vector<uint32_t> indices;
  std::vector<float> values;

  explicit SparseVector(uint32_t len) : indices(len), values(len) {}

  uint32_t length() const {
    assert(indices.size() == values.size());
    return indices.size();
  }
};

/** Returns the angle between two sparse vectors */
static float angle(const SparseVector& a, const SparseVector& b) {
  float total = 0, ma = 0, mb = 0;
  uint32_t ia = 0, ib = 0;
  while (ia < a.length() && ib < b.length()) {
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
  for (uint32_t i = 0; i < a.length(); i++) {
    ma += a.values[i] * a.values[i];
  }

  for (uint32_t i = 0; i < b.length(); i++) {
    mb += b.values[i] * b.values[i];
  }

  return std::acos(total / (std::sqrt(ma) * std::sqrt(mb)));
}

}  // namespace thirdai::hashing