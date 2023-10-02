#include "PatchSum.h"
#include <bolt/src/nn/autograd/Computation.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

std::string nextPatchSumOpName() {
  static uint32_t constructed = 0;
  return "patch_sum_" + std::to_string(++constructed);
}

PatchSum::PatchSum(size_t n_patches, size_t patch_dim)
    : Op(nextPatchSumOpName()), _n_patches(n_patches), _patch_dim(patch_dim) {}

void PatchSum::forward(const ComputationList& inputs, TensorPtr& output,
                       uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);

  const BoltVector& patches = inputs.at(0)->tensor()->getVector(index_in_batch);

  BoltVector& sum = output->getVector(index_in_batch);

  std::fill(sum.activations, sum.activations + sum.len, 0.0);

  if (patches.isDense()) {
    if (patches.len != _patch_dim * _n_patches) {
      throw std::invalid_argument("Expected input to patch dim to have " +
                                  std::to_string(_patch_dim * _n_patches) +
                                  " elements, but found " +
                                  std::to_string(patches.len) + " elements.");
    }
    for (size_t i = 0; i < _n_patches; i++) {
      for (size_t j = 0; j < _patch_dim; j++) {
        sum.activations[j] += patches.activations[i * _patch_dim + j];
      }
    }
  } else {
    for (size_t i = 0; i < patches.len; i++) {
      // If the patches are sparse then each should only have indices in the
      // range [0, patch_dim). Since the output of this op is dense we can just
      // index directly into the output using the index to accumulate the values. 
      if (patches.active_neurons[i] >= _patch_dim) {
        throw std::invalid_argument(
            "Cannot sum sparse index " +
            std::to_string(patches.active_neurons[i]) +
            " in PatchSum for patch_dim=" + std::to_string(_patch_dim) + ".");
      }
      sum.activations[patches.active_neurons[i]] += patches.activations[i];
    }
  }
}

void PatchSum::backpropagate(ComputationList& inputs, TensorPtr& output,
                             uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  assert(inputs.size() == 1);

  BoltVector& patches = inputs.at(0)->tensor()->getVector(index_in_batch);
  if (!patches.hasGradients()) {
    return;
  }

  const BoltVector& sum = output->getVector(index_in_batch);

  if (patches.isDense()) {
    if (patches.len != _patch_dim * _n_patches) {
      throw std::invalid_argument("Expected input to patch dim to have " +
                                  std::to_string(_patch_dim * _n_patches) +
                                  " elements, but found " +
                                  std::to_string(patches.len) + " elements.");
    }
    for (size_t i = 0; i < _n_patches; i++) {
      for (size_t j = 0; j < _patch_dim; j++) {
        patches.gradients[i * _patch_dim + j] += sum.gradients[j];
      }
    }
  } else {
    for (size_t i = 0; i < patches.len; i++) {
      if (patches.active_neurons[i] >= _patch_dim) {
        throw std::invalid_argument(
            "Cannot sum sparse index " +
            std::to_string(patches.active_neurons[i]) +
            " in PatchSum for patch_dim=" + std::to_string(_patch_dim) + ".");
      }
      patches.gradients[i] += sum.gradients[patches.active_neurons[i]];
    }
  }
}

void PatchSum::updateParameters(float learning_rate, uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

uint32_t PatchSum::dim() const { return _patch_dim; }

std::optional<uint32_t> PatchSum::nonzeros(const ComputationList& inputs,
                                           bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;
  return _patch_dim;
}

void PatchSum::initOptimizer() {}

void PatchSum::disableSparseParameterUpdates() {}

void PatchSum::enableSparseParameterUpdates() {}

std::vector<std::vector<float>*> PatchSum::gradients() { return {}; }

std::vector<std::vector<float>*> PatchSum::parameters() { return {}; }

void PatchSum::summary(std::ostream& summary, const ComputationList& inputs,
                       const Computation* output) const {
  summary << "PatchSum(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name() << "[n_patches=" << _n_patches
          << ", patch_dim=" << _patch_dim << "]";
}

void PatchSum::setSerializeOptimizer(bool should_serialize_optimizer) {
  (void)should_serialize_optimizer;
}

ComputationPtr PatchSum::apply(ComputationPtr input) {
  if (input->dim() != _patch_dim * _n_patches) {
    throw std::invalid_argument(
        "Cannot apply PatchSum expecting " + std::to_string(_n_patches) +
        " patches of dim " + std::to_string(_patch_dim) +
        " to input with dim " + std::to_string(input->dim()) + ".");
  }

  return Computation::make(shared_from_this(), {std::move(input)});
}

}  // namespace thirdai::bolt