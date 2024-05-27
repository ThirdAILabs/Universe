#include "AllocationManager.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt {

AllocationManager::AllocationManager(ComputationList computations)
    : _computations(std::move(computations)),
      _allocated_batch_size(0),
      _using_sparsity(true) {}

void AllocationManager::reallocateIfNeeded(uint32_t batch_size,
                                           bool use_sparsity) {
  if (batch_size == _allocated_batch_size && use_sparsity == _using_sparsity) {
    return;
  }

  for (auto& comp : _computations) {
    comp->allocate(batch_size, use_sparsity);
  }
  _allocated_batch_size = batch_size;
  _using_sparsity = use_sparsity;
}

void AllocationManager::resetOutputGradients(uint32_t index_in_batch) {
  for (auto& comp : _computations) {
    comp->tensor()->getVector(index_in_batch).zeroOutGradients();
  }
}

void AllocationManager::forceReallocation() {
  for (auto& comp : _computations) {
    comp->allocate(_allocated_batch_size, _using_sparsity);
  }
}

template void AllocationManager::serialize(cereal::BinaryInputArchive&);
template void AllocationManager::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void AllocationManager::serialize(Archive& archive) {
  // The allocated batch size should not be saved since state should be
  // reallocated when the model is reloaded. This is a dummy placeholder to
  // ensure ensure compatability since it was saved by mistake before.
  uint32_t ignore_allocated_batch_size = 0;
  archive(_computations, ignore_allocated_batch_size, _using_sparsity);
}

}  // namespace thirdai::bolt