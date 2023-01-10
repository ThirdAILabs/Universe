#include "InputTensor.h"
#include <optional>
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

std::shared_ptr<InputTensor> InputTensor::make(
    uint32_t dim, SparsityType sparsity_type,
    std::optional<uint32_t> num_nonzeros) {
  return std::make_shared<InputTensor>(dim, sparsity_type, num_nonzeros);
}

std::optional<uint32_t> InputTensor::numNonzeros(bool use_sparsity) const {
  (void)use_sparsity;  // Sparsity is fixed for inputs.
  return _num_nonzeros;
}

BoltVector& InputTensor::getVector(uint32_t index) {
  return (*_input_batch)[index];
}

void InputTensor::setInputs(const BoltBatch& batch) {
  std::optional<std::pair<uint32_t, uint32_t>> num_nonzeros_range =
      std::nullopt;
  if (_num_nonzeros) {
    num_nonzeros_range = {*_num_nonzeros, *_num_nonzeros};
  }
  batch.verifyExpectedDimension(dim(), num_nonzeros_range, "Input");

  _input_batch = const_cast<BoltBatch*>(&batch);
}

SparsityType InputTensor::sparsityType() const { return _sparsity_type; }

}  // namespace thirdai::bolt::nn::tensor