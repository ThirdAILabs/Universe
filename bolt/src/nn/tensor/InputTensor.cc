#include "InputTensor.h"

namespace thirdai::bolt::nn::tensor {

InputTensor::InputTensor(uint32_t dim, bool sparse,
                         std::optional<uint32_t> num_nonzeros)
    : Tensor(dim, sparse, num_nonzeros) {}

void InputTensor::setInputs(BoltBatch& batch) { _input_batch = &batch; }

BoltVector& InputTensor::getVector(uint32_t index) {
  return (*_input_batch)[index];
}

}  // namespace thirdai::bolt::nn::tensor