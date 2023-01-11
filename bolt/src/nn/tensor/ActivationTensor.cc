#include "ActivationTensor.h"
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>

namespace thirdai::bolt::nn::tensor {

std::string nextActivationTensorName() {
  static uint32_t constructed = 0;
  return "act_" + std::to_string(++constructed);
}

ActivationTensor::ActivationTensor(uint32_t dim, ops::OpPtr source,
                                   TensorList inputs)
    : Tensor(dim, nextActivationTensorName()),
      _source(std::move(source)),
      _inputs(std::move(inputs)) {}

std::shared_ptr<ActivationTensor> ActivationTensor::make(uint32_t dim,
                                                         ops::OpPtr source,
                                                         TensorList inputs) {
  return std::make_shared<ActivationTensor>(dim, std::move(source),
                                            std::move(inputs));
}

ops::OpPtr ActivationTensor::source() const { return _source; }

const TensorList& ActivationTensor::inputs() const { return _inputs; }

void ActivationTensor::forward(uint32_t index_in_batch, bool training) {
  _source->forward(_inputs, this, index_in_batch, training);
}

void ActivationTensor::backpropagate(uint32_t index_in_batch) {
  _source->backpropagate(_inputs, this, index_in_batch);
}

std::optional<uint32_t> ActivationTensor::numNonzeros(bool use_sparsity) const {
  return _source->numNonzerosInOutput(_inputs, use_sparsity);
}

BoltVector& ActivationTensor::getVector(uint32_t index) {
  return _vectors[index];
}

void ActivationTensor::allocate(uint32_t batch_size, bool use_sparsity) {
  _vectors.clear();
  _vectors.reserve(batch_size);

  uint32_t num_nonzeros = _source->numNonzerosInOutput(_inputs, use_sparsity);

  _activations.assign(batch_size * num_nonzeros, 0.0);
  _gradients.assign(batch_size * num_nonzeros, 0.0);

  if (use_sparsity && num_nonzeros < dim()) {
    _active_neurons.assign(batch_size * num_nonzeros, 0);

    for (uint32_t i = 0; i < batch_size; i++) {
      _vectors.emplace_back(_active_neurons.data() + i * num_nonzeros,
                            _activations.data() + i * num_nonzeros,
                            _gradients.data() + i * num_nonzeros, num_nonzeros);
    }
  } else {
    for (uint32_t i = 0; i < batch_size; i++) {
      _vectors.emplace_back(nullptr, _activations.data() + i * num_nonzeros,
                            _gradients.data() + i * num_nonzeros, num_nonzeros);
    }
  }
}

void ActivationTensor::addInput(InputTensorPtr input) {
  _inputs.push_back(std::move(input));
}

std::vector<uint32_t> ActivationTensor::shape() const {
  if (_vectors.empty()) {
    return {0, 0};
  }
  uint32_t batch_size = _vectors.size();
  return {batch_size, _vectors.at(0).len};
}

const uint32_t* ActivationTensor::activeNeuronsPtr() const {
  return _active_neurons.data();
}

const float* ActivationTensor::activationsPtr() const {
  return _activations.data();
}

const float* ActivationTensor::gradientsPtr() const {
  return _gradients.data();
}

ActivationTensorPtr asActivationTensor(const tensor::TensorPtr& tensor) {
  return std::dynamic_pointer_cast<ActivationTensor>(tensor);
}

}  // namespace thirdai::bolt::nn::tensor