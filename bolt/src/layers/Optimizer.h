#pragma once

#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <cstdint>
#include <vector>

namespace thirdai::bolt {

struct AdamOptimizer {
  explicit AdamOptimizer(uint64_t len)
      : gradients(len, 0.0), momentum(len, 0.0), velocity(len, 0.0) {}

  std::vector<float> gradients;
  std::vector<float> momentum;
  std::vector<float> velocity;

  // Cereal needs public empty constructor if it is wrapped around optional
  explicit AdamOptimizer(){};

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(gradients, momentum, velocity);
  }
};

}  // namespace thirdai::bolt