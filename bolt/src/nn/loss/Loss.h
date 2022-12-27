#pragma once

#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::nn::loss {

class Loss {
 public:
  virtual void computeGradients(uint32_t index_in_batch,
                                const BoltVector& label) = 0;

  virtual std::vector<tensor::ActivationTensorPtr> outputsUsed() const = 0;
};

using LossPtr = std::shared_ptr<Loss>;

}  // namespace thirdai::bolt::nn::loss