#include "FullyConnectedLayer.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <random>
#include <sstream>
#include <tuple>
#include <unordered_map>

namespace thirdai::bolt {

FullyConnectedLayer::FullyConnectedLayer(
    const FullyConnectedLayerConfig& config, uint64_t prev_dim)
    : _dim(config.dim),
      _prev_dim(prev_dim),
      _max_batch_size(0),
      _sparse_dim(config.sparsity * config.dim),
      _sparsity(config.sparsity),
      _act_func(config.act_func),
      _sampling_config(config.sampling_config),
      _force_sparse_for_inference(false),
      _is_restricted_class(false) {
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
  std::generate(_biases, _biases + _dim, [&]() { return dist(eng); });

  if (_sparsity < 1.0) {
    _hasher = new hashing::DWTAHashFunction(
        _prev_dim, _sampling_config.hashes_per_table,
        _sampling_config.num_tables, _sampling_config.range_pow);

    _hash_table = new hashtable::SampledHashTable<uint32_t>(
        _sampling_config.num_tables, _sampling_config.reservoir_size,
        1 << _sampling_config.range_pow);

    buildHashTables();

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

void FullyConnectedLayer::forward(const BoltVector& input, BoltVector& output,
                                  const uint32_t* labels, uint32_t label_len) {
  if (output.active_neurons == nullptr) {
    if (input.len == _prev_dim) {
      // TODO(Nicholas): Re-implement this case with dense matrix library
      forwardImpl<true, true>(input, output, labels, label_len);
    } else {
      forwardImpl<true, false>(input, output, labels, label_len);
    }
  } else {
    if (input.len == _prev_dim) {
      forwardImpl<false, true>(input, output, labels, label_len);
    } else {
      forwardImpl<false, false>(input, output, labels, label_len);
    }
  }
}

template <bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::forwardImpl(const BoltVector& input,
                                      BoltVector& output,
                                      const uint32_t* labels,
                                      uint32_t label_len) {
  selectActiveNeurons<DENSE, PREV_DENSE>(input, output, labels, label_len);

  float max_act = 0;
  uint32_t len_out = DENSE ? _dim : _sparse_dim;
  std::fill_n(output.gradients, len_out, 0);

  for (uint64_t n = 0; n < len_out; n++) {
    // Because DENSE is known at compile time the compiler can remove this
    // conditional
    uint64_t act_neuron = DENSE ? n : output.active_neurons[n];
    _is_active[act_neuron] = true;
    float act = _biases[act_neuron];
    for (uint64_t i = 0; i < input.len; i++) {
      // Because PREV_DENSE is known at compile time the compiler can remove
      // this conditional
      uint32_t prev_act_neuron = PREV_DENSE ? i : input.active_neurons[i];
      act += _weights[act_neuron * _prev_dim + prev_act_neuron] *
             input.activations[i];
    }
    switch (_act_func) {
      case ActivationFunc::ReLU:
        if (act < 0) {
          output.activations[n] = 0;
        } else {
          output.activations[n] = act;
        }
        break;
      case ActivationFunc::Softmax:
        output.activations[n] = act;
        if (max_act < act) {
          max_act = act;
        }
        break;
      case ActivationFunc::MeanSquared:
        output.activations[n] = act;
        break;
    }
  }

  if (_act_func == ActivationFunc::Softmax) {
    float total = 0;
    for (uint64_t n = 0; n < len_out; n++) {
      output.activations[n] = std::exp(output.activations[n] - max_act);
      total += output.activations[n];
    }
    for (uint64_t n = 0; n < len_out; n++) {
      output.activations[n] /= (total + EPS);
    }
  }
}

void FullyConnectedLayer::backpropagate(BoltVector& input, BoltVector& output) {
  if (output.active_neurons == nullptr) {
    if (input.len == _prev_dim) {
      backpropagateImpl<false, true, true>(input, output);
    } else {
      backpropagateImpl<false, true, false>(input, output);
    }
  } else {
    if (input.len == _prev_dim) {
      backpropagateImpl<false, false, true>(input, output);
    } else {
      backpropagateImpl<false, false, false>(input, output);
    }
  }
}

void FullyConnectedLayer::backpropagateInputLayer(BoltVector& input,
                                                  BoltVector& output) {
  if (output.active_neurons == nullptr) {
    if (input.len == _prev_dim) {
      backpropagateImpl<true, true, true>(input, output);
    } else {
      backpropagateImpl<true, true, false>(input, output);
    }
  } else {
    if (input.len == _prev_dim) {
      backpropagateImpl<true, false, true>(input, output);
    } else {
      backpropagateImpl<true, false, false>(input, output);
    }
  }
}

constexpr float FullyConnectedLayer::actFuncDerivative(float x) {
  switch (_act_func) {
    case ActivationFunc::ReLU:
      return x > 0 ? 1.0 : 0.0;
    case ActivationFunc::Softmax:
      // return 1.0; // Commented out because Clang tidy doesn't like
      // consecutive identical branches
    case ActivationFunc::MeanSquared:
      return 1.0;
      // default:
      //   return 0.0;
  }
  // This is impossible to reach, but the compiler gave a warning saying it
  // reached the end of a non void function without it.
  return 0.0;
}

template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::backpropagateImpl(BoltVector& input,
                                            BoltVector& output) {
  uint32_t len_out = DENSE ? _dim : _sparse_dim;

  for (uint64_t n = 0; n < len_out; n++) {
    output.gradients[n] *= actFuncDerivative(output.activations[n]);
    // Because DENSE is known at compile time the compiler can remove this
    // conditional
    uint32_t act_neuron = DENSE ? n : output.active_neurons[n];
    for (uint64_t i = 0; i < input.len; i++) {
      // Because PREV_DENSE is known at compile time the compiler can remove
      // this conditional
      uint32_t prev_act_neuron = PREV_DENSE ? i : input.active_neurons[i];
      _w_gradient[act_neuron * _prev_dim + prev_act_neuron] +=
          output.gradients[n] * input.activations[i];
      if (!FIRST_LAYER) {
        input.gradients[i] +=
            output.gradients[n] *
            _weights[act_neuron * _prev_dim + prev_act_neuron];
      }
    }
    _b_gradient[act_neuron] += output.gradients[n];
  }
}

template <bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::selectActiveNeurons(const BoltVector& input,
                                              BoltVector& output,
                                              const uint32_t* labels,
                                              uint32_t label_len) {
  
  if(_is_restricted_class){
     uint32_t counter = 0;
     for (size_t i = 0; i < _restricted_class_len; i++){
        output.active_neurons[counter++] = _restricted_class[i]; 
     }
     return;
  }

  if (DENSE) {
    return;
  }

  std::unordered_set<uint32_t> active_set;

  for (uint32_t i = 0; i < label_len; i++) {
    active_set.insert(labels[i]);
  }

  uint32_t* hashes = new uint32_t[_hash_table->numTables()];
  if (PREV_DENSE) {
    _hasher->hashSingleDense(input.activations, input.len, hashes);
  } else {
    _hasher->hashSingleSparse(input.active_neurons, input.activations,
                              input.len, hashes);
  }

  if (_force_sparse_for_inference && _act_func == ActivationFunc::Softmax) {
    _hash_table->queryAndInsertForInference(hashes, active_set, _sparse_dim);
  } else {
    _hash_table->queryBySet(hashes, active_set);
  }

  delete[] hashes;

  if (active_set.size() < _sparse_dim) {
    uint32_t rand_offset = rand() % _dim;
    while (active_set.size() < _sparse_dim) {
      active_set.insert(_rand_neurons[rand_offset++]);
      rand_offset = rand_offset % _dim;
    }
  }

  uint32_t cnt = 0;
  for (uint32_t i = 0; i < label_len; i++) {
    if (cnt >= _sparse_dim) {
      break;
    }
    output.active_neurons[cnt++] = labels[i];
    active_set.erase(labels[i]);
  }

  for (auto x : active_set) {
    if (cnt >= _sparse_dim) {
      break;
    }
    output.active_neurons[cnt++] = x;
  }
}

void FullyConnectedLayer::updateParameters(float lr, uint32_t iter, float B1,
                                           float B2, float eps) {
  float B1_bias_corrected = static_cast<float>(1 - pow(B1, iter));
  float B2_bias_corrected = static_cast<float>(1 - pow(B2, iter));

#pragma omp parallel for default(none) \
    shared(lr, B1, B1_bias_corrected, B2, B2_bias_corrected, eps)
  for (uint64_t n = 0; n < _dim; n++) {
    if (!_is_active[n]) {
      continue;
    }

    for (uint64_t i = 0; i < _prev_dim; i++) {
      auto indx = n * _prev_dim + i;
      float grad = _w_gradient[indx];
      _w_momentum[indx] = B1 * _w_momentum[indx] + (1 - B1) * grad;
      _w_velocity[indx] = B2 * _w_velocity[indx] + (1 - B2) * grad * grad;

      _weights[indx] +=
          lr * (_w_momentum[indx] / B1_bias_corrected) /
          (std::sqrt(_w_velocity[indx] / B2_bias_corrected) + eps);

      _w_gradient[indx] = 0;
    }

    float grad = _b_gradient[n];
    _b_momentum[n] = B1 * _b_momentum[n] + (1 - B1) * grad;
    _b_velocity[n] = B2 * _b_velocity[n] + (1 - B2) * grad * grad;

    _biases[n] += lr * (_b_momentum[n] / B1_bias_corrected) /
                  (std::sqrt(_b_velocity[n] / B2_bias_corrected) + eps);

    _b_gradient[n] = 0;
    _is_active[n] = false;
  }
}

void FullyConnectedLayer::buildHashTables() {
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

void FullyConnectedLayer::reBuildHashFunction() {
  if (_sparsity >= 1.0) {
    return;
  }
  delete _hasher;

  _hasher = new hashing::DWTAHashFunction(
      _prev_dim, _sampling_config.hashes_per_table, _sampling_config.num_tables,
      _sampling_config.range_pow);
}

void FullyConnectedLayer::shuffleRandNeurons() {
  if (_sparsity < 1.0) {
    std::shuffle(_rand_neurons, _rand_neurons + _dim, std::random_device{});
  }
}

float* FullyConnectedLayer::getWeights() {
  float* weights_copy = new float[_dim * _prev_dim];
  std::copy(_weights, _weights + _dim * _prev_dim, weights_copy);

  return weights_copy;
}

float* FullyConnectedLayer::getBiases() {
  float* biases_copy = new float[_dim];
  std::copy(_biases, _biases + _dim, biases_copy);

  return biases_copy;
}


void FullyConnectedLayer::restrictClass(uint32_t* class_ids, uint32_t class_ids_len) {
     if(_act_func == ActivationFunc::Softmax)
     {
        _is_restricted_class = true;
        _sparsity = class_ids_len/_dim;
        _sparse_dim = class_ids_len;
        _restricted_class_len = class_ids_len;
        _restricted_class = new uint32_t[_restricted_class_len];
        
        for (size_t i = 0; i < _restricted_class_len; i++)
        {
          _restricted_class[i] = class_ids[i];
        }
        
     }
}


FullyConnectedLayer::~FullyConnectedLayer() {
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

  delete[] _restricted_class;
}

}  // namespace thirdai::bolt