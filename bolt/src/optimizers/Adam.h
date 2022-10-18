#pragma once

#include <bolt/src/optimizers/Optimizer.h>
#include <cmath>

namespace thirdai::bolt {

class AdamOptimizer final : public Optimizer {
 public:
  AdamOptimizer(std::vector<float>& parameters, std::vector<float>& gradients,
                float beta1 = 0.9, float beta2 = 0.999)
      : Optimizer(parameters, gradients), _beta1(beta1), _beta2(beta2) {
    // This will initialize the bias corrected version of beta1 and beta2.
    completeTrainStep();

    _momentum.assign(_parameter_length, 0.0);
    _velocity.assign(_parameter_length, 0.0);
  }

  void updateRange(uint64_t start, uint64_t length, float learning_rate,
                   bool parallel) final {
    if (parallel) {
#pragma omp parallel for default(none) shared(start, length, learning_rate)
      for (uint64_t i = start; i < start + length; i++) {
        updateAtIndex(i, learning_rate);
      }
    } else {
      for (uint64_t i = start; i < start + length; i++) {
        updateAtIndex(i, learning_rate);
      }
    }
  }

  void updateAtIndex(uint64_t index, float learning_rate) final {
    assert(index < _parameter_length);

    float grad = _gradients[index];
    _gradients[index] = 0;
    assert(!std::isnan(grad));

    _momentum[index] = _beta1 * _momentum[index] + (1 - _beta1) * grad;
    _velocity[index] = _beta2 * _velocity[index] + (1 - _beta2) * grad * grad;
    assert(!std::isnan(_momentum[index]));
    assert(!std::isnan(_velocity[index]));

    _parameters[index] +=
        learning_rate * (_momentum[index] / _beta1_corrected) /
        (std::sqrt(_velocity[index] / _beta2_corrected) + eps);
    assert(!std::isnan(_parameters[index]));
  }

  void completeTrainStep() final {
    ++_iter;

    _beta1_corrected = static_cast<float>(1 - pow(_beta1, _iter));
    _beta2_corrected = static_cast<float>(1 - pow(_beta2, _iter));
  }

 private:
  static constexpr float eps = 0.0000001;

  std::vector<float> _momentum;
  std::vector<float> _velocity;

  float _beta1;
  float _beta1_corrected;
  float _beta2;
  float _beta2_corrected;

  uint32_t _iter;
};

}  // namespace thirdai::bolt