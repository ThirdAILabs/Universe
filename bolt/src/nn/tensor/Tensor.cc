#include "Tensor.h"
#include <bolt_vector/src/BoltVector.h>
#include <optional>
#include <stdexcept>

namespace thirdai::bolt::nn::tensor {

Tensor::Tensor(uint32_t batch_size, uint32_t dim, uint32_t nonzeros)
    : _dim(dim), _nonzeros(nonzeros) {
  if (nonzeros < dim) {
    _active_neurons.assign(batch_size * nonzeros, 0);
  }

  _activations.assign(batch_size * nonzeros, 0.0);
  _gradients.assign(batch_size * nonzeros, 0.0);

  _vectors.reserve(batch_size);

  for (uint32_t i = 0; i < batch_size * nonzeros; i += nonzeros) {
    uint32_t* active_neurons = nullptr;
    if (nonzeros < dim) {
      active_neurons = _active_neurons.data() + i;
    }

    _vectors.emplace_back(active_neurons, _activations.data() + i,
                          _gradients.data() + i, nonzeros);
  }
}

Tensor::Tensor(BoltBatch&& batch, uint32_t dim)
    : _dim(dim), _nonzeros(std::nullopt) {
  if (batch.getBatchSize() == 0) {
    throw std::invalid_argument("Cannot convert empty batch to tensor.");
  }

  bool is_dense = batch.begin()->isDense();

  for (const auto& vec : batch) {
    if (vec.isDense() != is_dense) {
      throw std::invalid_argument(
          "All vectors in batch must have same sparsity to convert to tensor.");
    }
    if (vec.hasGradients()) {
      throw std::invalid_argument(
          "Cannot convert vector with gradients to tensor.");
    }
    if (is_dense && vec.len != _dim) {
      throw std::invalid_argument(
          "All dense vectors must have the same length to convert to tensor.");
    }

    for (uint32_t i = 0; i < vec.len; i++) {
      if (!is_dense) {
        if (vec.active_neurons[i] >= dim) {
          throw std::invalid_argument(
              "Found sparse index " + std::to_string(vec.active_neurons[i]) +
              " that exceeded dimension " + std::to_string(dim) + ".");
        }
        _active_neurons.push_back(vec.active_neurons[i]);
      }
      _activations.push_back(vec.activations[i]);
    }
  }

  uint32_t offset = 0;
  for (const auto& vec : batch) {
    uint32_t* active_neurons =
        is_dense ? nullptr : _active_neurons.data() + offset;

    _vectors.emplace_back(active_neurons, _activations.data() + offset, nullptr,
                          vec.len);
    offset += vec.len;
  }
}

std::shared_ptr<Tensor> Tensor::dense(uint32_t batch_size, uint32_t dim) {
  return std::make_shared<Tensor>(batch_size, dim, dim);
}

std::shared_ptr<Tensor> Tensor::sparse(uint32_t batch_size, uint32_t dim,
                                       uint32_t nonzeros) {
  return std::make_shared<Tensor>(batch_size, dim, nonzeros);
}

std::shared_ptr<Tensor> Tensor::convert(BoltBatch&& batch, uint32_t dim) {
  return std::make_shared<Tensor>(std::move(batch), dim);
}

uint32_t Tensor::dim() const { return _dim; }

std::optional<uint32_t> Tensor::nonzeros() const { return _nonzeros; }

BoltVector& Tensor::getVector(uint32_t index) { return _vectors[index]; }

uint32_t Tensor::batchSize() const { return _vectors.size(); }

const uint32_t* Tensor::activeNeuronsPtr() const {
  return _active_neurons.empty() ? nullptr : _active_neurons.data();
}

const float* Tensor::activationsPtr() const { return _activations.data(); }

const float* Tensor::gradientsPtr() const {
  return _gradients.empty() ? nullptr : _gradients.data();
}

}  // namespace thirdai::bolt::nn::tensor