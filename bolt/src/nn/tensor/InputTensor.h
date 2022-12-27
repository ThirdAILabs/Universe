#pragma once

#include "Tensor.h"

namespace thirdai::bolt::nn::tensor {

class InputTensor final : public Tensor {
 public:
  InputTensor(uint32_t dim, bool sparse, std::optional<uint32_t> num_nonzeros)
      : Tensor(dim, sparse, num_nonzeros) {}

  void setInputs(BoltBatch& batch) { _input_batch = &batch; }

  BoltVector& getVector(uint32_t index) final { return (*_input_batch)[index]; }

 private:
  BoltBatch* _input_batch;
};

using InputTensorPtr = std::shared_ptr<InputTensor>;

}  // namespace thirdai::bolt::nn::tensor