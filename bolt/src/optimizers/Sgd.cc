#include "Sgd.h"

namespace thirdai::bolt::optimizers {

void Sgd::updateRange(uint64_t start, uint64_t length, float learning_rate,
                      bool parallel) {
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

void Sgd::updateAtIndex(uint64_t index, float learning_rate) {
  _parameters[index] += learning_rate * _gradients[index];
  _gradients[index] = 0;
}

void Sgd::completeTrainStep() {}

}  // namespace thirdai::bolt::optimizers