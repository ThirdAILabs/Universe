#include "AllocationManager.h"
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::model {

AllocationManager::AllocationManager(autograd::ComputationList computations)
    : _computations(std::move(computations)),
      _allocated_batch_size(0),
      _current_batch_size(0),
      _using_sparsity(true) {}

void AllocationManager::reallocateForBatch(uint32_t batch_size,
                                           bool use_sparsity) {
  _current_batch_size = batch_size;
  if (batch_size <= _allocated_batch_size && use_sparsity == _using_sparsity) {
    return;
  }

  for (auto& comp : _computations) {
    comp->allocate(batch_size, use_sparsity);
  }
}

void AllocationManager::resetOutputGradients(uint32_t index_in_batch) {
  for (auto& comp : _computations) {
    comp->tensor()->getVector(index_in_batch).zeroOutGradients();
  }
}

}  // namespace thirdai::bolt::nn::model