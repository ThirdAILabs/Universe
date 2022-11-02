#include "SignedMomentum.h"
#include <algorithm>

namespace thirdai::bolt::optimizers {

void SignedMomentum::updateRange(uint64_t start, uint64_t length,
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

void SignedMomentum::updateAtIndex(uint64_t index, float learning_rate) {
  float gradient = clip(_gradients[index], /*min=*/-1e2, /*max=*/1e2);

  if ((gradient < 0.0 && !_last_gradient_positive[index]) ||
      (gradient > 0.0 && _last_gradient_positive[index])) {
    float scale_factor = _learning_rate_scaling_factor[index];
    scale_factor = clip(scale_factor * _increase_scale_factor, /*min=*/1e-8,
                        /*max=*/1e2);
    _learning_rate_scaling_factor[index] = scale_factor;
  } else if (gradient != 0.0) {
    /**
     * If the gradient is 0 it is we do not update the scaling factor or
     * recorded sign because no update is being applied. This means that the
     * sign is always for the last nonzero update, and the scaling factor is
     * only updated for nonzero gradients. We are likely to get a fair number of
     * zero gradients due to sparsity and ReLU.
     */
    float scale_factor = _learning_rate_scaling_factor[index];
    scale_factor =
        clip(scale_factor * _decrease_scale_factor, /*min=*/1e-8, /*max=*/1e2);
    _learning_rate_scaling_factor[index] = scale_factor;

    _last_gradient_positive[index] = gradient > 0.0;
  }

  _parameters[index] +=
      learning_rate * _learning_rate_scaling_factor[index] * gradient;

  _gradients[index] = 0.0;
}

void SignedMomentum::completeTrainStep() {}

}  // namespace thirdai::bolt::optimizers