#include "AllocationManager.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::model {

AllocationManager::AllocationManager(autograd::ComputationList computations)
    : _computations(std::move(computations)),
      _allocated_batch_size(0),
      _using_sparsity(true) {}

void AllocationManager::reallocateIfNeeded(uint32_t batch_size,
                                           bool use_sparsity) {
  if (batch_size <= _allocated_batch_size && use_sparsity == _using_sparsity) {
    return;
  }

  for (auto& comp : _computations) {
    comp->allocate(batch_size, use_sparsity);
  }
}

void AllocationManager::resetOutputGradients(uint32_t index_in_batch) {
  for (auto& comp : _computations) {
    auto& tensor = comp->tensor();

    uint32_t start = tensor->rangeStart(index_in_batch);
    uint32_t end = tensor->rangeEnd(index_in_batch);

    for (uint32_t i = start; i < end; i++) {
      tensor->getVector(i).zeroOutGradients();
    }
  }
}

template void AllocationManager::serialize(cereal::BinaryInputArchive&);
template void AllocationManager::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void AllocationManager::serialize(Archive& archive) {
  archive(_computations, _allocated_batch_size, _using_sparsity);
}

}  // namespace thirdai::bolt::nn::model