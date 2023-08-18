#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt {

/**
 * This class manages the allocation of Tensors in the model. This means
 * that it is responsible for tracking the currenly allocated batch size and
 * whether sparsity is used so that the tensors can be reallocated when either
 * changes. It can also be called to reallocate the tensors when the sparsity of
 * one of the ops changes. Essentially this acts as a cache of the last used
 * batch size and sparsity to avoid reallocating unless necessary.
 */
class AllocationManager {
 public:
  explicit AllocationManager(ComputationList computations);

  AllocationManager() : _allocated_batch_size(0), _using_sparsity(false) {}

  /**
   * This method will call the allocate(...) method of each computation if the
   * provided batch size is greater than the currently allocated batch size or
   * if whether or not sparsity is being used is changing.
   */
  void reallocateIfNeeded(uint32_t batch_size, bool use_sparsity);

  /**
   * Sets all of the gradients to 0 for the ith vector of the output tensors.
   * This should be called before executing the logic in backpropagate in the
   * model.
   */
  void resetOutputGradients(uint32_t index_in_batch);

  /**
   * Reallocates all of the state for the current batch size. Used when sparsity
   * is changed in the model.
   */
  void forceReallocation();

 private:
  ComputationList _computations;

  uint32_t _allocated_batch_size;

  bool _using_sparsity;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::bolt