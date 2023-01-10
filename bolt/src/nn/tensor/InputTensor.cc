#include "InputTensor.h"
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::bolt::nn::tensor {

std::string nextInputTensorName() {
  static uint32_t constructed = 0;
  return "input_" + std::to_string(++constructed);
}

InputTensor::InputTensor(uint32_t dim, std::optional<uint32_t> num_nonzeros)
    : Tensor(dim, nextInputTensorName()), _num_nonzeros(num_nonzeros) {}

std::shared_ptr<InputTensor> InputTensor::make(
    uint32_t dim, std::optional<uint32_t> num_nonzeros) {
  return std::make_shared<InputTensor>(dim, num_nonzeros);
}

std::optional<uint32_t> InputTensor::numNonzeros(bool use_sparsity) const {
  (void)use_sparsity;  // Sparsity is fixed for inputs.
  return _num_nonzeros;
}

BoltVector& InputTensor::getVector(uint32_t index) {
  return (*_input_batch)[index];
}

void InputTensor::setInputs(const BoltBatch& batch) {
  batch.verifyExpectedDimension(dim(), /* num_nonzeros_range= */ std::nullopt,
                                "Input");
  if (_num_nonzeros) {
    for (const auto& vec : batch) {
      if (!vec.isDense() && vec.len != *_num_nonzeros) {
        std::stringstream error;
        error << "Expected sparse input to have " << *_num_nonzeros
              << " nonzeros but found vector with " << vec.len << " nonzeros.";
        throw std::invalid_argument(error.str());
      }
    }
  }

  _input_batch = const_cast<BoltBatch*>(&batch);
}

}  // namespace thirdai::bolt::nn::tensor