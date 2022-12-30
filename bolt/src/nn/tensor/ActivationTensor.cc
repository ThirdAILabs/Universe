#include "ActivationTensor.h"
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::tensor {

std::string nextActivationTensorName() {
  static uint32_t constructed = 0;
  return "act_" + std::to_string(++constructed);
}

ActivationTensor::ActivationTensor(uint32_t dim, uint32_t sparse_nonzeros,
                                   ops::Op* source)
    : Tensor(dim, nextActivationTensorName()),
      _sparse_nonzeros(sparse_nonzeros),
      _using_sparsity(true),
      _source(source) {}

std::shared_ptr<ActivationTensor> ActivationTensor::make(
    uint32_t dim, uint32_t sparse_nonzeros, ops::Op* source) {
  return std::make_shared<ActivationTensor>(dim, sparse_nonzeros, source);
}

std::optional<uint32_t> ActivationTensor::numNonzeros() const {
  if (_using_sparsity) {
    return _sparse_nonzeros;
  }
  return dim();
}

BoltVector& ActivationTensor::getVector(uint32_t index) {
  return _vectors[index];
}

void ActivationTensor::allocate(uint32_t batch_size, bool use_sparsity) {
  _vectors.clear();
  _vectors.reserve(batch_size);
  _using_sparsity = use_sparsity;

  uint32_t num_nonzeros = _using_sparsity ? _sparse_nonzeros : dim();

  _activations.assign(batch_size * num_nonzeros, 0.0);
  _gradients.assign(batch_size * num_nonzeros, 0.0);

  if (use_sparsity && _sparse_nonzeros < dim()) {
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

void ActivationTensor::updateSparsity(uint32_t new_sparse_nonzeros) {
  _sparse_nonzeros = new_sparse_nonzeros;

  allocate(_vectors.size(), _using_sparsity);

  for (auto& dependant : _dependant_ops) {
    dependant->notifyInputSparsityChange();
  }
}

ops::Op* ActivationTensor::source() const { return _source; }

std::vector<uint32_t> ActivationTensor::shape() const {
  uint32_t batch_size = _vectors.size();
  return {batch_size, dim()};
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

}  // namespace thirdai::bolt::nn::tensor