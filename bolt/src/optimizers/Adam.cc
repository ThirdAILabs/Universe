#include "Adam.h"
#include <cassert>

namespace thirdai::bolt::optimizers {

void AdamOptimizer::updateRange(uint64_t start, uint64_t length,
                                float learning_rate, bool parallel) {
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

void AdamOptimizer::updateAtIndex(uint64_t index, float learning_rate) {
  assert(index < _parameter_length);

  float grad = _gradients[index];
  _gradients[index] = 0;
  assert(!std::isnan(grad));

  _momentum[index] = _beta1 * _momentum[index] + (1 - _beta1) * grad;
  _velocity[index] = _beta2 * _velocity[index] + (1 - _beta2) * grad * grad;
  assert(!std::isnan(_momentum[index]));
  assert(!std::isnan(_velocity[index]));

  _parameters[index] += learning_rate * (_momentum[index] / _beta1_corrected) /
                        (std::sqrt(_velocity[index] / _beta2_corrected) + eps);
  assert(!std::isnan(_parameters[index]));
}

void AdamOptimizer::completeTrainStep() {
  ++_iter;

  // These terms are used to correct the bias in adam, we compute them here
  // avoid having to recompute them for each call to one of the update methods
  // in a given batch.
  _beta1_corrected = static_cast<float>(1 - pow(_beta1, _iter));
  _beta2_corrected = static_cast<float>(1 - pow(_beta2, _iter));
}

}  // namespace thirdai::bolt::optimizers