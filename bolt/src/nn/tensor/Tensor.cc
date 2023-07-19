#include "Tensor.h"
#include <bolt_vector/src/BoltVector.h>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace thirdai::bolt::nn::tensor {

Tensor::Tensor(uint32_t batch_size, uint32_t dim, uint32_t nonzeros)
    : _dim(dim), _nonzeros(nonzeros) {
  if (nonzeros == 0) {
    throw std::invalid_argument("Cannot allocate tensor with 0 nonzeros.");
  }

  if (nonzeros < dim) {
    _active_neurons.assign(batch_size * nonzeros, 0);
  }

  _activations.assign(batch_size * nonzeros, 0.0);
  _gradients.assign(batch_size * nonzeros, 0.0);

  _vectors.reserve(batch_size);

  for (uint32_t offset = 0; offset < batch_size * nonzeros;
       offset += nonzeros) {
    uint32_t* active_neurons = nullptr;
    if (nonzeros < dim) {
      active_neurons = _active_neurons.data() + offset;
    }

    _vectors.emplace_back(
        /* active_neurons= */ active_neurons,
        /* activations= */ _activations.data() + offset,
        /* gradients= */ _gradients.data() + offset, /* len= */ nonzeros);
  }
}

Tensor::Tensor(const BoltBatch& batch, uint32_t dim)
    : _dim(dim), _nonzeros(std::nullopt) {
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
                          /* activations= */ _activations.data() + offset,
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

std::shared_ptr<Tensor> Tensor::dense(uint32_t batch_size, uint32_t dim) {
  return std::make_shared<Tensor>(/* batch_size= */ batch_size, /* dim= */ dim,
                                  /* nonzeros= */ dim);
}

std::shared_ptr<Tensor> Tensor::sparse(uint32_t batch_size, uint32_t dim,
                                       uint32_t nonzeros) {
  return std::make_shared<Tensor>(/* batch_size= */ batch_size, /* dim= */ dim,
                                  /* nonzeros= */ nonzeros);
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

uint32_t Tensor::dim() const { return _dim; }

std::optional<uint32_t> Tensor::nonzeros() const { return _nonzeros; }

BoltVector& Tensor::getVector(uint32_t index) {
  assert(index < _vectors.size());
  return _vectors[index];
}

uint32_t Tensor::batchSize() const { return _vectors.size(); }

const uint32_t* Tensor::activeNeuronsPtr() const {
  return _active_neurons.empty() ? nullptr : _active_neurons.data();
}

const float* Tensor::activationsPtr() const {
  return _activations.empty() ? nullptr : _activations.data();
}

std::pair<std::vector<uint32_t>, std::vector<float> >
Tensor::topKIndexValuePair(uint32_t topk) {
  std::vector<float> topk_activations;
  std::vector<uint32_t> topk_active_neurons;

  uint32_t batch_size = batchSize();
  uint32_t total_size = batch_size * topk;

  topk_activations.resize(total_size);
  topk_active_neurons.resize(total_size);

  for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    int idx_ = topk - 1;
    TopKActivationsQueue topk_activations_queue =
        getVector(batch_idx).findKLargestActivations(topk);

    while (!topk_activations_queue.empty() && idx_ >= 0) {
      ValueIndexPair val_idx_pair = topk_activations_queue.top();

      topk_activations[batch_idx * topk + idx_] = val_idx_pair.first;
      topk_active_neurons[batch_idx * topk + idx_] = val_idx_pair.second;

      topk_activations_queue.pop();
      idx_--;
    }
  }
  return std::make_pair(topk_active_neurons, topk_activations);
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
      }
    }
  }
}

}  // namespace thirdai::bolt::nn::tensor