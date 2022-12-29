#include "InputTensor.h"
#include <string>

namespace thirdai::bolt::nn::tensor {

std::string nextInputTensorName() {
  static uint32_t constructed = 0;
  return "input_" + std::to_string(++constructed);
}

InputTensor::InputTensor(uint32_t dim, SparsityType sparsity_type,
                         std::optional<uint32_t> num_nonzeros)
    : Tensor(dim, nextInputTensorName()),
      _num_nonzeros(num_nonzeros),
      _sparsity_type(sparsity_type) {}

std::optional<uint32_t> InputTensor::numNonzeros() const {
  return _num_nonzeros;
}

BoltVector& InputTensor::getVector(uint32_t index) {
  return (*_input_batch)[index];
}

void InputTensor::setInputs(BoltBatch& batch) { _input_batch = &batch; }

SparsityType InputTensor::sparsityType() const { return _sparsity_type; }

}  // namespace thirdai::bolt::nn::tensor