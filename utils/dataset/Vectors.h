#pragma once

#include <cstdint>

namespace thirdai::utils {

struct SparseVector {
  uint32_t* indices;
  float* values;
  uint32_t len;
};

struct DenseVector {
  float* values;
  uint32_t dim;
};

};  // namespace thirdai::utils