#pragma once

#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>

namespace thirdai::bolt::nn::ops {

class Op {
 public:
  virtual void forward(uint32_t index_in_batch) = 0;

  virtual void backpropagate(uint32_t index_in_batch) = 0;

  virtual void updateParameters(float learning_rate) = 0;

  virtual void initOptimizer() = 0;

  virtual void disableSparseParameterUpdates() = 0;

  virtual const std::vector<tensor::ActivationTensorPtr>& outputs() const = 0;

  virtual void notifyInputSparsityChange() = 0;
};

}  // namespace thirdai::bolt::nn::ops