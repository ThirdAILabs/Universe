#include "Tensor.h"
#include <bolt_vector/src/BoltVector.h>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace thirdai::bolt::nn::tensor {

Tensor::Tensor(Dims dims, uint32_t nonzeros, bool with_grad)
    : _dims(std::move(dims)), _nonzeros(nonzeros) {
  if (nonzeros == 0) {
    throw std::invalid_argument("Cannot allocate tensor with 0 nonzeros.");
  }

  if (nonzeros < dim) {
    _active_neurons.assign(batch_size * nonzeros, 0);
  }

  uint32_t num_vectors =
      std::reduce(_dims.begin(), _dims.end() - 1, 1, std::multiplies<>());

  uint32_t inner_dim_3d =
      std::reduce(_dims.begin() + 1, _dims.end() - 1, 1, std::multiplies<>());

  _dims_2d = {num_vectors, _dims.back()};
  _dims_3d = {_dims.front(), inner_dim_3d, _dims.back()};

  bool sparse = nonzeros < _dims.back();

  if (sparse) {
    _indices.assign(num_vectors * nonzeros, 0);
  } else if (nonzeros > _dims.back()) {
    throw std::invalid_argument(
        "The number of nonzeros cannot be larger than the final dimension.");
  }

  _values.assign(num_vectors * nonzeros, 0.0);
  if (with_grad) {
    _gradients.assign(num_vectors * nonzeros, 0.0);
  }

  _vectors.reserve(num_vectors);

  for (uint32_t offset = 0; offset < num_vectors * nonzeros;
       offset += nonzeros) {
    uint32_t* active_neurons = nullptr;
    if (sparse) {
      active_neurons = _indices.data() + offset;
    }

    float* gradients = nullptr;
    if (with_grad) {
      gradients = _gradients.data() + offset;
    }

    _vectors.emplace_back(
        /* active_neurons= */ active_neurons,
        /* activations= */ _values.data() + offset,
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
    std::copy(indices, indices + _indices.size(), _indices.begin());
  }
  std::copy(values, values + _values.size(), _values.begin());
}

Tensor::Tensor(const BoltBatch& batch, uint32_t dim)
    : _dims({batch.getBatchSize(), dim}),
      _nonzeros(std::nullopt),
      _dims_2d({batch.getBatchSize(), dim}),
      _dims_3d({batch.getBatchSize(), 1, dim}) {
  if (batch.getBatchSize() == 0) {
    throw std::invalid_argument("Cannot convert empty batch to tensor.");
  }

  checkBatchContents(batch, _dim);

  uint64_t total_dim = 0;
  for (const auto& vec : batch) {
    total_dim += vec.len;
  }

  bool is_sparse = !batch.begin()->isDense();

  if (is_sparse) {
    _active_neurons.reserve(total_dim);
  }
  _activations.reserve(total_dim);
  _vectors.reserve(batch.getBatchSize());

  for (const auto& vec : batch) {
    if (is_sparse) {
      _active_neurons.insert(_active_neurons.end(), vec.active_neurons,
                             vec.active_neurons + vec.len);
    }
    _activations.insert(_activations.end(), vec.activations,
                        vec.activations + vec.len);
  }

  uint32_t offset = 0;
  for (const auto& vec : batch) {
    uint32_t* active_neurons =
        is_sparse ? _active_neurons.data() + offset : nullptr;

    _vectors.emplace_back(/* active_neurons= */ active_neurons,
                          /* activations= */ _values.data() + offset,
                          /* gradients= */ nullptr, /* len= */ vec.len);
    offset += vec.len;
  }
}

Tensor::Tensor(BoltBatch&& batch, uint32_t dim)
    : _dim(dim), _nonzeros(std::nullopt) {
  checkBatchContents(batch, _dim);

  // NOLINTNEXTLINE clang tidy wants this in the intitializer list.
  _vectors = std::move(batch.vectors());

  for (auto& vec : _vectors) {
    if (!vec.ownsMemory()) {
      vec = vec.copy();
    }
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

std::shared_ptr<Tensor> Tensor::dense(uint32_t batch_size, uint32_t dim) {
  return std::make_shared<Tensor>(/* batch_size= */ batch_size,
                                  /* dim= */ dim,
                                  /* nonzeros= */ dim);
}

std::shared_ptr<Tensor> Tensor::fromArray(const uint32_t* indices,
                                          const float* values,
                                          tensor::Dims dims, uint32_t nonzeros,
                                          bool with_grad) {
  return std::make_shared<Tensor>(indices, values, std::move(dims), nonzeros,
                                  with_grad);
}

std::shared_ptr<Tensor> Tensor::copy(const BoltBatch& batch, uint32_t dim) {
  return std::make_shared<Tensor>(batch, dim);
}

std::shared_ptr<Tensor> Tensor::convert(BoltBatch&& batch, uint32_t dim) {
  return std::make_shared<Tensor>(std::move(batch), dim);
}

std::shared_ptr<Tensor> Tensor::convert(BoltVector&& vector, uint32_t dim) {
  BoltBatch batch({std::move(vector)});
  return convert(std::move(batch), dim);
}

BoltVector& Tensor::index2d(uint32_t i) {
  assert(i < _dims_2d.at(0));
  return _vectors[i];
}

BoltVector& Tensor::index3d(uint32_t i, uint32_t j) {
  assert(i < _dims_3d.at(0) && j < _dims_3d.at(1));
  return _vectors[i * _dims_3d.at(1) + j];
}

uint32_t* Tensor::indicesAtIndex3d(uint32_t i) {
  assert(index_in_batch < batchSize());
  if (!_nonzeros) {
    throw std::runtime_error("Cannot access sub array of ragged tensor.");
  }
  if (_indices.empty()) {
    throw std::runtime_error("Cannot access indices of dense tensor.");
  }
  return _indices.data() + i * _dims_3d.at(1) * (*_nonzeros);
}

float* Tensor::valuesAtIndex3d(uint32_t i) {
  assert(index_in_batch < batchSize());
  if (!_nonzeros) {
    throw std::runtime_error("Cannot access sub array of ragged tensor.");
  }
  return _values.data() + i * _dims_3d.at(1) * (*_nonzeros);
}

float* Tensor::gradientsAtIndex3d(uint32_t i) {
  assert(index_in_batch < batchSize());
  if (!_nonzeros) {
    throw std::runtime_error("Cannot access sub array of ragged tensor.");
  }
  if (_gradients.empty()) {
    throw std::runtime_error(
        "Cannot access gradients of a tensor without grad.");
  }
  return _gradients.data() + i * _dims_3d.at(1) * (*_nonzeros);
}

const uint32_t* Tensor::indicesPtr() const {
  return _indices.empty() ? nullptr : _indices.data();
}

const float* Tensor::valuesPtr() const {
  return _values.empty() ? nullptr : _activations.data();
}

const float* Tensor::gradientsPtr() const {
  return _gradients.empty() ? nullptr : _gradients.data();
}

void Tensor::checkBatchContents(const BoltBatch& batch, uint32_t dim) {
  if (batch.getBatchSize() == 0) {
    throw std::invalid_argument("Cannot convert empty batch to tensor.");
  }

  bool is_dense = batch.begin()->isDense();

  for (const auto& vec : batch) {
    if (vec.len == 0) {
      throw std::invalid_argument("Cannot convert empty vector to tensor.");
    }
    if (vec.isDense() != is_dense) {
      throw std::invalid_argument(
          "All vectors in batch must have same sparsity to convert to "
          "tensor.");
    }
    // Since this constructor is intended to be used to convert inputs we
    // don't need to handle the case where there are gradients.
    if (vec.hasGradients()) {
      throw std::invalid_argument(
          "Cannot convert vector with gradients to tensor.");
    }
    if (is_dense && vec.len != dim) {
      throw std::invalid_argument(
          "All dense vectors must have the same length to convert to "
          "tensor.");
    }

    for (uint32_t i = 0; i < vec.len; i++) {
      if (!is_dense) {
        if (vec.active_neurons[i] >= dim) {
          throw std::invalid_argument(
              "Found sparse index " + std::to_string(vec.active_neurons[i]) +
              " that exceeded dimension " + std::to_string(dim) + ".");
        }
      }
    }
  }
}

}  // namespace thirdai::bolt::nn::tensor