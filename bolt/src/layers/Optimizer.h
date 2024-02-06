#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace thirdai::bolt {

constexpr float BETA1 = 0.9;
constexpr float BETA2 = 0.999;
constexpr float EPS = 0.0000001;

struct AdamOptimizer {
  explicit AdamOptimizer(uint64_t len)
      : gradients(len, 0.0), momentum(len, 0.0), velocity(len, 0.0) {}

  std::vector<float> gradients;
  std::vector<float> momentum;
  std::vector<float> velocity;

  // Cereal needs public empty constructor if it is wrapped around optional
  explicit AdamOptimizer() {}

  void applyUpdate(std::vector<float>& params, float learning_rate,
                   uint32_t train_steps) {
    assert(params.size() == gradients.size());

    float B1_bias_corrected = static_cast<float>(1 - pow(BETA1, train_steps));
    float B2_bias_corrected = static_cast<float>(1 - pow(BETA2, train_steps));

#pragma omp parallel for default(none) \
    shared(params, B1_bias_corrected, B2_bias_corrected, learning_rate)
    for (uint64_t n = 0; n < params.size(); n++) {
      float grad = gradients[n];
      assert(!std::isnan(grad));

      momentum[n] = BETA1 * momentum[n] + (1 - BETA1) * grad;
      velocity[n] = BETA2 * velocity[n] + (1 - BETA2) * grad * grad;
      assert(!std::isnan(momentum[n]));
      assert(!std::isnan(velocity[n]));

      // params[n] += learning_rate * (momentum[n] / B1_bias_corrected) /
      //              (std::sqrt(velocity[n] / B2_bias_corrected) + EPS);

      if (grad > 0) {
        params[n] += learning_rate;
      } else {
        params[n] -= learning_rate;
      }

      assert(!std::isnan(params[n]));

      gradients[n] = 0;

      (void)B1_bias_corrected;
      (void)B2_bias_corrected;
    }
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(momentum, gradients, velocity);
  }
};

}  // namespace thirdai::bolt