#pragma once

#include <cstdint>
#include <vector>

namespace thirdai::bolt {

struct AdamOptimizer {
  explicit AdamOptimizer(uint64_t len)
      : gradients(len, 0.0), momentum(len, 0.0), velocity(len, 0.0) {}

  std::vector<float> gradients;
  std::vector<float> momentum;
  std::vector<float> velocity;
};

}  // namespace thirdai::bolt