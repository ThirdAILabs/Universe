#include "ActivationTensor.h"
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::tensor {

ActivationTensor::ActivationTensor(uint32_t dim, uint32_t num_nonzeros)
    : Tensor(dim, num_nonzeros < dim, num_nonzeros), _using_sparsity(false) {}

BoltVector& ActivationTensor::getVector(uint32_t index) {
  return _vectors[index];
}

void ActivationTensor::allocate(uint32_t batch_size, bool use_sparsity) {
  _vectors.clear();
  _vectors.reserve(batch_size);
  _using_sparsity = use_sparsity;

  uint32_t num_nonzeros = _num_nonzeros.value();

  _activations.assign(batch_size * num_nonzeros, 0.0);
  _gradients.assign(batch_size * num_nonzeros, 0.0);
  if (use_sparsity && _sparse) {
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

void ActivationTensor::updateSparsity(uint32_t num_nonzeros) {
  _num_nonzeros = num_nonzeros;
  _sparse = num_nonzeros < dim();

  allocate(_vectors.size(), _using_sparsity);

  for (auto& dependant : _dependant_ops) {
    dependant->notifyInputSparsityChange();
  }
}

}  // namespace thirdai::bolt::nn::tensor