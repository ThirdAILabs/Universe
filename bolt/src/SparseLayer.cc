#include "SparseLayer.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

namespace thirdai::bolt {

SparseLayer::SparseLayer(uint64_t dim, uint64_t prev_dim, float sparsity,
                         ActivationFunc act_func,
                         SamplingConfig sampling_config)
    : _dim(dim),
      _prev_dim(prev_dim),
      _batch_size(0),
      _sparse_dim(sparsity * dim),
      _sparsity(sparsity),
      _act_func(act_func),
      _active_lens(nullptr),
      _active_neurons(nullptr),
      _activations(nullptr),
      _errors(nullptr),
      _sampling_config(sampling_config) {
  uint64_t total_size = _dim * _prev_dim;

  _weights = new float[total_size];
  _w_gradient = new float[total_size]();
  _w_momentum = new float[total_size]();
  _w_velocity = new float[total_size]();

  _biases = new float[_dim];
  _b_gradient = new float[_dim]();
  _b_momentum = new float[_dim]();
  _b_velocity = new float[_dim]();

  _is_active = new bool[_dim]();  // TODO(nicholas): bitvector?

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(_weights, _weights + total_size, [&]() { return dist(eng); });
  std::generate(_biases, _biases + dim, [&]() { return dist(eng); });

  if (_sparsity < 1.0) {
    _hasher = new utils::DWTAHashFunction(
        _prev_dim, _sampling_config.hashes_per_table, _sampling_config.num_tables,
        _sampling_config.range_pow);

    _hash_table = new utils::SampledHashTable<uint32_t>(
        _sampling_config.num_tables, _sampling_config.reservoir_size,
        _sampling_config.range_pow);

    BuildHashTables();

    _rand_neurons = new uint32_t[_dim];
    for (uint32_t i = 0; i < _dim; i++) {
      _rand_neurons[i] = i;
    }

    std::shuffle(_rand_neurons, _rand_neurons + _dim, rd);
  } else {
    _hasher = nullptr;
    _hash_table = nullptr;
    _rand_neurons = nullptr;
  }
}

void SparseLayer::FeedForward(uint32_t batch_indx, const uint32_t* indices,
                              const float* values, uint32_t len,
                              uint32_t* labels, uint32_t label_len) {
  SelectActiveNeurons(batch_indx, indices, values, len, labels, label_len);

  float max_act = 0;
  for (uint64_t n = 0; n < _active_lens[batch_indx]; n++) {
    uint64_t act_neuron = _active_neurons[batch_indx][n];
    _is_active[act_neuron] = true;
    float act = _biases[act_neuron];
    for (uint64_t i = 0; i < len; i++) {
      act += _weights[act_neuron * _prev_dim + indices[i]] * values[i];
    }
    switch (_act_func) {
      case ActivationFunc::ReLU:
        if (act < 0) {
          _activations[batch_indx][n] = 0;
        } else {
          _activations[batch_indx][n] = act;
        }
        break;
      case ActivationFunc::Softmax:
        _activations[batch_indx][n] = act;
        if (max_act < act) {
          max_act = act;
        }
        break;
    }
  }

  if (_act_func == ActivationFunc::Softmax) {
    float total = 0;
    for (uint64_t n = 0; n < _active_lens[batch_indx]; n++) {
      _activations[batch_indx][n] =
          std::exp(_activations[batch_indx][n] - max_act);
      total += _activations[batch_indx][n];
    }
    for (uint64_t n = 0; n < _active_lens[batch_indx]; n++) {
      _activations[batch_indx][n] /= (total + EPS);
    }
  }
}

constexpr float SparseLayer::ActFuncDerivative(float x) {
  switch (_act_func) {
    case ActivationFunc::ReLU:
      return x > 0 ? 1.0 : 0.0;
    case ActivationFunc::Softmax:
      return 1.0;
  }
  // This is impossible to reach, but the compiler gave a warning saying it
  // reached the end of a non void function wihtout it.
  return 0.0;
}

template void SparseLayer::BackPropagateImpl<true>(uint32_t, const uint32_t*,
                                                   const float*, float*,
                                                   uint32_t);

template void SparseLayer::BackPropagateImpl<false>(uint32_t, const uint32_t*,
                                                    const float*, float*,
                                                    uint32_t);

template <bool FIRST_LAYER>
void SparseLayer::BackPropagateImpl(uint32_t batch_indx,
                                    const uint32_t* indices,
                                    const float* values, float* prev_errors,
                                    uint32_t len) {
  for (uint64_t n = 0; n < _active_lens[batch_indx]; n++) {
    _errors[batch_indx][n] *= ActFuncDerivative(_activations[batch_indx][n]);
    for (uint64_t i = 0; i < len; i++) {
      _w_gradient[_active_neurons[batch_indx][n] * _prev_dim + indices[i]] +=
          _errors[batch_indx][n] * values[i];
      if (!FIRST_LAYER) {
        prev_errors[i] +=
            _errors[batch_indx][n] *
            _weights[_active_neurons[batch_indx][n] * _prev_dim + indices[i]];
      }
    }
    _b_gradient[_active_neurons[batch_indx][n]] += _errors[batch_indx][n];
  }
}

void SparseLayer::ComputeErrors(uint32_t batch_indx, const uint32_t* labels,
                                uint32_t label_len) {
  float frac = 1.0 / label_len;

  for (uint64_t n = 0; n < _active_lens[batch_indx]; n++) {
    if (std::find(labels, labels + label_len, _active_neurons[batch_indx][n]) !=
        labels + label_len) {
      _errors[batch_indx][n] = (frac - _activations[batch_indx][n]) / _batch_size;
    } else {
      _errors[batch_indx][n] = -_activations[batch_indx][n] / _batch_size;
    }
  }
}

void SparseLayer::SelectActiveNeurons(uint32_t batch_indx,
                                      const uint32_t* indices,
                                      const float* values, uint32_t len,
                                      uint32_t* labels, uint32_t label_len) {
  if (_sparsity == 1.0) {
    _active_lens[batch_indx] = _dim;
    for (uint32_t i = 0; i < _dim; i++) {
      _active_neurons[batch_indx][i] = i;
    }
  } else {
    std::unordered_set<uint32_t> active_set;

    for (uint32_t i = 0; i < label_len; i++) {
      active_set.insert(labels[i]);
    }

    uint32_t* hashes = new uint32_t[_hash_table->numTables()];
    _hasher->hashSingleSparse(indices, values, len, hashes);
    _hash_table->queryBySet(hashes, active_set);
    delete[] hashes;

    if (active_set.size() < _sparse_dim) {
      uint32_t rand_offset = rand() % _dim;
      while (active_set.size() < _sparse_dim) {
        active_set.insert(_rand_neurons[rand_offset++]);
        rand_offset = rand_offset % _dim;
      }
    }

    uint32_t active_len = _sparse_dim;
    _active_lens[batch_indx] = active_len;

    uint32_t cnt = 0;
    for (uint32_t i = 0; i < label_len; i++) {
      if (cnt >= _sparse_dim) {
        break;
      }
      _active_neurons[batch_indx][cnt++] = labels[i];
      active_set.erase(labels[i]);
    }
    for (auto x : active_set) {
      if (cnt >= _sparse_dim) {
        break;
      }
      _active_neurons[batch_indx][cnt++] = x;
    }
  }
  std::fill_n(_errors[batch_indx], _dim, 0);
}

void SparseLayer::UpdateParameters(float lr, uint32_t iter, float B1, float B2,
                                   float eps) {
  float B1_ = static_cast<float>(1 - pow(B1, iter));
  float B2_ = static_cast<float>(1 - pow(B2, iter));

#pragma omp parallel for default(none) shared(lr, B1, B1_, B2, B2_, eps)
  for (uint64_t n = 0; n < _dim; n++) {
    if (!_is_active[n]) {
      continue;
    }

    for (uint64_t i = 0; i < _prev_dim; i++) {
      auto indx = n * _prev_dim + i;
      float grad = _w_gradient[indx];
      _w_momentum[indx] = B1 * _w_momentum[indx] + (1 - B1) * grad;
      _w_velocity[indx] = B2 * _w_velocity[indx] + (1 - B2) * grad * grad;

      _weights[indx] += lr * (_w_momentum[indx] / B1_) /
                       (std::sqrt(_w_velocity[indx] / B2_) + eps);

      _w_gradient[indx] = 0;
    }

    float grad = _b_gradient[n];
    _b_momentum[n] = B1 * _b_momentum[n] + (1 - B1) * grad;
    _b_velocity[n] = B2 * _b_velocity[n] + (1 - B2) * grad * grad;

    _biases[n] +=
        lr * (_b_momentum[n] / B1_) / (std::sqrt(_b_velocity[n] / B2_) + eps);

    _b_gradient[n] = 0;
    _is_active[n] = false;
  }
}

void SparseLayer::BuildHashTables() {
  if (_sparsity >= 1.0) {
    return;
  }
  uint64_t num_tables = _hash_table->numTables();
  // TODO(nicholas): hashes could be array with size max(batch size, dim) that
  // is allocated once
  uint32_t* hashes = new uint32_t[num_tables * _dim];

#pragma omp parallel for default(none) shared(num_tables, hashes)
  for (uint64_t n = 0; n < _dim; n++) {
    _hasher->hashSingleDense(_weights + n * _prev_dim, _prev_dim,
                            hashes + n * num_tables);
  }

  _hash_table->clearTables();
  _hash_table->insertSequential(_dim, 0, hashes);

  delete[] hashes;
}

void SparseLayer::ReBuildHashFunction() {
  if (_sparsity >= 1.0) {
    return;
  }
  delete _hasher;

  _hasher = new utils::DWTAHashFunction(
      _prev_dim, _sampling_config.hashes_per_table, _sampling_config.num_tables,
      _sampling_config.range_pow);
}

void SparseLayer::SetBatchSize(uint64_t new_batch_size) {
  if (new_batch_size <= _batch_size) {
    return;
  }

  for (uint64_t batch = 0; batch < _batch_size; batch++) {
    delete[] _active_neurons[batch];
    delete[] _activations[batch];
    delete[] _errors[batch];
  }

  delete[] _active_lens;
  delete[] _active_neurons;
  delete[] _activations;
  delete[] _errors;

  _batch_size = new_batch_size;

  _active_lens = new uint32_t[_batch_size];
  _active_neurons = new uint32_t*[_batch_size];
  _activations = new float*[_batch_size];
  _errors = new float*[_batch_size];

  for (uint64_t batch = 0; batch < _batch_size; batch++) {
    _active_neurons[batch] = new uint32_t[_dim];
    _activations[batch] = new float[_dim];
    _errors[batch] = new float[_dim]();
  }
}

void SparseLayer::SetSparsity(float new_sparsity) {
  _sparsity = new_sparsity;
  _sparse_dim = _sparsity * _dim;
}

void SparseLayer::ShuffleRandNeurons() {
  if (_sparsity < 1.0) {
    std::shuffle(_rand_neurons, _rand_neurons + _dim, std::random_device{});
  }
}

float* SparseLayer::GetWeights() {
  float* weights_copy = new float[_dim * _prev_dim];
  std::copy(_weights, _weights + _dim * _prev_dim, weights_copy);

  return weights_copy;
}

float* SparseLayer::GetBiases() {
  float* biases_copy = new float[_dim];
  std::copy(_biases, _biases + _dim, biases_copy);

  return biases_copy;
}

SparseLayer::~SparseLayer() {
  for (uint64_t batch = 0; batch < _batch_size; batch++) {
    delete[] _active_neurons[batch];
    delete[] _activations[batch];
    delete[] _errors[batch];
  }

  delete[] _active_lens;
  delete[] _active_neurons;
  delete[] _activations;
  delete[] _errors;

  delete[] _weights;
  delete[] _w_gradient;
  delete[] _w_momentum;
  delete[] _w_velocity;

  delete[] _biases;
  delete[] _b_gradient;
  delete[] _b_momentum;
  delete[] _b_velocity;

  delete[] _is_active;

  delete _hasher;
  delete _hash_table;
  delete[] _rand_neurons;
}

}  // namespace thirdai::bolt
