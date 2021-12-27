#include "FullyConnectedLayer.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <immintrin.h>
#include <random>
#include <sstream>
#include <tuple>
#include <unordered_map>

// #define USE_VECTORIZATION

namespace thirdai::bolt {

FullyConnectedLayer::FullyConnectedLayer(
    const FullyConnectedLayerConfig& config, uint64_t prev_dim)
    : _dim(config.dim),
      _prev_dim(prev_dim),
      _max_batch_size(0),
      _sparse_dim(config.sparsity * config.dim),
      _sparsity(config.sparsity),
      _act_func(config.act_func),
      _sampling_config(config.sampling_config) {
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

void FullyConnectedLayer::forward(const VectorState& input, VectorState& output,
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
void FullyConnectedLayer::forwardImpl(const VectorState& input,
                                      VectorState& output,
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

void FullyConnectedLayer::backpropagate(VectorState& input,
                                        VectorState& output) {
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

void FullyConnectedLayer::backpropagateInputLayer(VectorState& input,
                                                  VectorState& output) {
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
void FullyConnectedLayer::backpropagateImpl(VectorState& input,
                                            VectorState& output) {
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
void FullyConnectedLayer::selectActiveNeurons(const VectorState& input,
                                              VectorState& output,
                                              const uint32_t* labels,
                                              uint32_t label_len) {
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
  _hash_table->queryBySet(hashes, active_set);
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
  float B1_ = static_cast<float>(1 - pow(B1, iter));
  float B2_ = static_cast<float>(1 - pow(B2, iter));

#if defined __SSE__ && defined USE_VECTORIZATION
  float one_minus_B1 = 1 - B1;
  float one_minus_B2 = 1 - B2;
  float lr_b1 = lr / B1_;
#endif

  // #pragma omp parallel for default(none) shared(lr, B1, B1_, B2, B2_, eps)
  for (uint64_t n = 0; n < _dim; n++) {
    if (!_is_active[n]) {
      continue;
    }

#if defined __SSE__ && defined USE_VECTORIZATION
    for (uint64_t i = 0; i < (_prev_dim & (~3)); i += 4) {
      auto index = n * _prev_dim + i;

      __m128 grad = _mm_load_ps(_w_gradient + index);

      __m128 mom, vel;
      {  // _w_momentum[indx] = B1 * _w_momentum[indx] + (1 - B1) * grad;
        __m128 omB1 = _mm_load1_ps(&one_minus_B1);
        __m128 term2 = _mm_mul_ps(omB1, grad);

        __m128 B1_128 = _mm_load1_ps(&B1);
        mom = _mm_load_ps(_w_momentum + index);
        __m128 term1 = _mm_mul_ps(B1_128, mom);

        mom = _mm_add_ps(term1, term2);
        _mm_store_ps(_w_momentum + index, mom);
      }

      {  // _w_velocity[indx] = B2 * _w_velocity[indx] + (1 - B2) * grad * grad;
        __m128 omB2 = _mm_load1_ps(&one_minus_B2);
        __m128 grad_square = _mm_mul_ps(grad, grad);
        __m128 term2 = _mm_mul_ps(omB2, grad_square);

        __m128 B2_128 = _mm_load1_ps(&B2);
        vel = _mm_load_ps(_w_velocity + index);
        __m128 term1 = _mm_mul_ps(B2_128, vel);

        vel = _mm_add_ps(term1, term2);
        _mm_store_ps(_w_velocity + index, vel);
      }

      {  // _weights[indx] += lr * (_w_momentum[indx] / B1_) /
         //                   (std::sqrt(_w_velocity[indx] / B2_) + eps);
        __m128 lr_b1_128 = _mm_load1_ps(&lr_b1);
        __m128 mom_term = _mm_mul_ps(lr_b1_128, mom);

        __m128 B2_hat = _mm_load1_ps(&B2_);
        __m128 vel_term = _mm_div_ps(vel, B2_hat);
        __m128 eps128 = _mm_load1_ps(&eps);
        vel_term = _mm_add_ps(vel_term, eps128);

        vel_term = _mm_rsqrt_ps(vel_term);

        __m128 new_weight = _mm_mul_ps(mom_term, vel_term);
        _mm_store_ps(_weights + index, new_weight);
      }

      {  // _w_gradient[indx] = 0;
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) {
          _w_gradient[index + j] = 0;
        }
      }
    }
    for (uint64_t i = (_prev_dim & (~3)); i < _prev_dim; i++) {
      auto indx = n * _prev_dim + i;
      float grad = _w_gradient[indx];
      _w_momentum[indx] = B1 * _w_momentum[indx] + (1 - B1) * grad;
      _w_velocity[indx] = B2 * _w_velocity[indx] + (1 - B2) * grad * grad;

      _weights[indx] += lr * (_w_momentum[indx] / B1_) /
                        (std::sqrt(_w_velocity[indx] / B2_) + eps);

      _w_gradient[indx] = 0;
    }
#else
#pragma GCC ivdep
    for (uint64_t i = 0; i < _prev_dim; i++) {
      auto indx = n * _prev_dim + i;
      float grad = _w_gradient[indx];
      _w_momentum[indx] = B1 * _w_momentum[indx] + (1 - B1) * grad;
      _w_velocity[indx] = B2 * _w_velocity[indx] + (1 - B2) * grad * grad;

      _weights[indx] += lr * (_w_momentum[indx] / B1_) /
                        (std::sqrt(_w_velocity[indx] / B2_) + eps);

      _w_gradient[indx] = 0;
    }
#endif

    float grad = _b_gradient[n];
    _b_momentum[n] = B1 * _b_momentum[n] + (1 - B1) * grad;
    _b_velocity[n] = B2 * _b_velocity[n] + (1 - B2) * grad * grad;

    _biases[n] +=
        lr * (_b_momentum[n] / B1_) / (std::sqrt(_b_velocity[n] / B2_) + eps);

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
}

}  // namespace thirdai::bolt