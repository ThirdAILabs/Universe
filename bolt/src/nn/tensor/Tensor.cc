#include "Tensor.h"
#include <bolt_vector/src/BoltVector.h>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace thirdai::bolt::nn::tensor {

Tensor::Tensor(Dims dims, uint32_t nonzeros, bool with_grad)
    : _dims(std::move(dims)), _nonzeros(nonzeros) {
  uint32_t num_vectors =
      std::reduce(_dims.begin(), _dims.end() - 1, 1, std::multiplies<>());

  _inner_dim_3d =
      std::reduce(_dims.begin() + 1, _dims.end() - 1, 1, std::multiplies<>());

  bool sparse = nonzeros < _dims.back();

  if (sparse) {
    _active_neurons.assign(num_vectors * nonzeros, 0);
  } else if (nonzeros > _dims.back()) {
    throw std::invalid_argument(
        "The number of nonzeros cannot be larger than the final dimension.");
  }

  _activations.assign(num_vectors * nonzeros, 0.0);
  if (with_grad) {
    _gradients.assign(num_vectors * nonzeros, 0.0);
  }

  _vectors.reserve(num_vectors);

  for (uint32_t offset = 0; offset < num_vectors * nonzeros;
       offset += nonzeros) {
    uint32_t* active_neurons = nullptr;
    if (sparse) {
      active_neurons = _active_neurons.data() + offset;
    }

    float* gradients = nullptr;
    if (with_grad) {
      gradients = _gradients.data() + offset;
    }

    _vectors.emplace_back(
        /* active_neurons= */ active_neurons,
        /* activations= */ _activations.data() + offset,
        /* gradients= */ gradients, /* len= */ nonzeros);
  }
}

Tensor::Tensor(const uint32_t* indices, const float* values, tensor::Dims dims,
               uint32_t nonzeros, bool with_grad)
    : Tensor(std::move(dims), nonzeros, with_grad) {
  if (isSparse() && !indices) {
    throw std::invalid_argument(
        "Must specify tensor indices if nonzeros is less than the last "
        "dimension.");
  }

  if (!isSparse() && indices) {
    throw std::invalid_argument("Cannot specify indices for a dense tensor.");
  }

  if (isSparse()) {
    std::copy(indices, indices + _active_neurons.size(),
              _active_neurons.begin());
  }

  if (values) {
    std::copy(values, values + _activations.size(), _activations.begin());
  } else {
    std::fill(_activations.begin(), _activations.end(), 1.0);
  }
}

Tensor::Tensor(const BoltBatch& batch, uint32_t dim)
    : _dims({batch.getBatchSize(), dim}),
      _nonzeros(std::nullopt),
      _inner_dim_3d(1) {
  if (batch.getBatchSize() == 0) {
    throw std::invalid_argument("Cannot convert empty batch to tensor.");
  }

  bool is_dense = batch.begin()->isDense();

  for (const auto& vec : batch) {
    if (vec.isDense() != is_dense) {
      throw std::invalid_argument(
          "All vectors in batch must have same sparsity to convert to tensor.");
    }
    // Since this constructor is intended to be used to convert inputs we don't
    // need to handle the case where there are gradients.
    if (vec.hasGradients()) {
      throw std::invalid_argument(
          "Cannot convert vector with gradients to tensor.");
    }
    if (is_dense && vec.len != dim) {
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

    _vectors.emplace_back(/* active_neurons= */ active_neurons,
                          /* activations= */ _activations.data() + offset,
                          /* gradients= */ nullptr, /* len= */ vec.len);
    offset += vec.len;
  }
}

std::shared_ptr<Tensor> Tensor::dense(Dims dims) {
  return std::make_shared<Tensor>(/* dims= */ dims,
                                  /* nonzeros= */ dims.back(),
                                  /* with_grad= */ true);
}

std::shared_ptr<Tensor> Tensor::sparse(Dims dims, uint32_t nonzeros) {
  return std::make_shared<Tensor>(/* dims= */ std::move(dims),
                                  /* nonzeros= */ nonzeros,
                                  /* with_grad= */ true);
}

std::shared_ptr<Tensor> Tensor::fromArray(const uint32_t* indices,
                                          const float* values,
                                          tensor::Dims dims, uint32_t nonzeros,
                                          bool with_grad) {
  return std::make_shared<Tensor>(indices, values, std::move(dims), nonzeros,
                                  with_grad);
}

std::shared_ptr<Tensor> Tensor::convert(const BoltBatch& batch, uint32_t dim) {
  return std::make_shared<Tensor>(batch, dim);
}

std::shared_ptr<Tensor> Tensor::convert(const BoltVector& vector,
                                        uint32_t dim) {
  BoltBatch batch({vector});
  return convert(batch, dim);
}

const Dims& Tensor::dims() const { return _dims; }

std::optional<uint32_t> Tensor::nonzeros() const { return _nonzeros; }

bool Tensor::isSparse() const { return !_active_neurons.empty(); }

BoltVector& Tensor::getVector(uint32_t index) {
  assert(index < _vectors.size());
  return _vectors[index];
}

uint32_t Tensor::batchSize() const { return _dims.front(); }

const uint32_t* Tensor::activeNeuronsPtr() const {
  return _active_neurons.empty() ? nullptr : _active_neurons.data();
}

const float* Tensor::activationsPtr() const { return _activations.data(); }

const float* Tensor::gradientsPtr() const {
  return _gradients.empty() ? nullptr : _gradients.data();
}

}  // namespace thirdai::bolt::nn::tensor