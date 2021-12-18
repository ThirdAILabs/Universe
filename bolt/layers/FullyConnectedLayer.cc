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

void FullyConnectedLayer::forward(const uint32_t* indices_in,
                                  const float* values_in, uint32_t len_in,
                                  uint32_t* indices_out, float* values_out,
                                  const uint32_t* labels, uint32_t label_len) {
  if (indices_out != nullptr) {
    if (len_in == _prev_dim) {
      // TODO(Nicholas): Re-implement this case with dense matrix library
      forwardImpl<true, true>(indices_in, values_in, len_in, indices_out,
                              values_out, labels, label_len);
    } else {
      forwardImpl<true, false>(indices_in, values_in, len_in, indices_out,
                               values_out, labels, label_len);
    }
  } else {
    if (len_in == _prev_dim) {
      forwardImpl<false, true>(indices_in, values_in, len_in, indices_out,
                               values_out, labels, label_len);
    } else {
      forwardImpl<false, false>(indices_in, values_in, len_in, indices_out,
                                values_out, labels, label_len);
    }
  }
}

template <bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::forwardImpl(const uint32_t* indices_in,
                                      const float* values_in, uint32_t len_in,
                                      uint32_t* indices_out, float* values_out,
                                      const uint32_t* labels,
                                      uint32_t label_len) {
  selectActiveNeurons<DENSE, PREV_DENSE>(batch_indx, indices, values, len,
                                         indices_out labels, label_len);
  std::fill_n(gradients_out, _dim, 0);

  float max_act = 0;
  uint32_t len_out = DENSE ? _dim : _sparse_dim;
  for (uint64_t n = 0; n < len_out; n++) {
    // Because DENSE is known at compile time the compiler can remove this
    // conditional
    uint64_t act_neuron = DENSE ? n : indices_out[n];
    _is_active[act_neuron] = true;
    float act = _biases[act_neuron];
    for (uint64_t i = 0; i < len_in; i++) {
      // Because PREV_DENSE is known at compile time the compiler can remove
      // this conditional
      uint32_t prev_act_neuron = PREV_DENSE ? i : indices_in[i];
      act += _weights[act_neuron * _prev_dim + prev_act_neuron] * values_in[i];
    }
    switch (_act_func) {
      case ActivationFunc::ReLU:
        if (act < 0) {
          values_out[n] = 0;
        } else {
          values_out[n] = act;
        }
        break;
      case ActivationFunc::Softmax:
        values_out[n] = act;
        if (max_act < act) {
          max_act = act;
        }
        break;
      case ActivationFunc::MeanSquared:
        values_out[n] = act;
    }
  }

  if (_act_func == ActivationFunc::Softmax) {
    float total = 0;
    for (uint64_t n = 0; n < len_out; n++) {
      values_out[n] = std::exp(values_out[n] - max_act);
      total += values_out[n];
    }
    for (uint64_t n = 0; n < len_out; n++) {
      values_out[n] /= (total + EPS);
    }
  }
}

void FullyConnectedLayer::backpropagate(const uint32_t* indices_in,
                                        const float* values_in,
                                        float* gradients_in, uint32_t len_in,
                                        const uint32_t* indices_out,
                                        const float* values_out,
                                        const float* gradients_out) {
  if (indices_out == nullptr) {
    if (len_in == _prev_dim) {
      backpropagateImpl<false, true, true>(indices_in, values_in, gradients_in,
                                           len_in, indices_out, values_out,
                                           gradients_out);
    } else {
      backpropagateImpl<false, true, false>(indices_in, values_in, gradients_in,
                                            len_in, indices_out, values_out,
                                            gradients_out);
    }
  } else {
    if (len_in == _prev_dim) {
      backpropagateImpl<false, false, true>(indices_in, values_in, gradients_in,
                                            len_in, indices_out, values_out,
                                            gradients_out);
    } else {
      backpropagateImpl<false, false, false>(indices_in, values_in,
                                             gradients_in, len_in, indices_out,
                                             values_out, gradients_out);
    }
  }
}

void FullyConnectedLayer::backpropagateInputLayer(const uint32_t* indices_in,
                                                  const float* values_in,
                                                  uint32_t len_in,
                                                  const uint32_t* indices_out,
                                                  const float* values_out,
                                                  const float* gradients_out) {
  if (indices_out == nullptr) {
    if (len_in == _prev_dim) {
      backpropagateImpl<true, true, true>(indices_in, values_in, nullptr,
                                          len_in, indices_out, values_out,
                                          gradients_out);
    } else {
      backpropagateImpl<true, true, false>(indices_in, values_in, nullptr,
                                           len_in, indices_out, values_out,
                                           gradients_out);
    }
  } else {
    if (len_in == _prev_dim) {
      backpropagateImpl<true, false, true>(indices_in, values_in, nullptr,
                                           len_in, indices_out, values_out,
                                           gradients_out);
    } else {
      backpropagateImpl<true, false, false>(indices_in, values_in, nullptr,
                                            len_in, indices_out, values_out,
                                            gradients_out);
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
  // reached the end of a non void function wihtout it.
  return 0.0;
}

template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::backpropagateImpl(
    const uint32_t* indices_in, const float* values_in, float* gradients_in,
    uint32_t len_in, const uint32_t* indices_out, const float* values_out,
    const float* gradients_out) {
  uint32_t len_out = DENSE ? _dim : _sparse_dim;

  for (uint64_t n = 0; n < len_out; n++) {
    _errors[batch_indx][n] *= actFuncDerivative(values_out[n]);
    // Because DENSE is known at compile time the compiler can remove this
    // conditional
    uint32_t act_neuron = DENSE ? n : indices_out[n];
    for (uint64_t i = 0; i < len_in; i++) {
      // Because PREV_DENSE is known at compile time the compiler can remove
      // this conditional
      uint32_t prev_act_neuron = PREV_DENSE ? i : indices_in[i];
      _w_gradient[act_neuron * _prev_dim + prev_act_neuron] +=
          gradients_out[n] * values_in[i];
      if (!FIRST_LAYER) {
        gradients_in[i] += gradients_out[n] *
                           _weights[act_neuron * _prev_dim + prev_act_neuron];
      }
    }
    _b_gradient[act_neuron] += gradients_out[n];
  }
}

// void FullyConnectedLayer::computeSoftmaxErrors(uint32_t batch_indx,
//                                                uint32_t batch_size,
//                                                const uint32_t* labels,
//                                                uint32_t label_len) {
//   if (_sparse_dim == _dim) {
//     computeSoftmaxErrorsImpl<true>(batch_indx, batch_size, labels,
//     label_len);
//   } else {
//     computeSoftmaxErrorsImpl<false>(batch_indx, batch_size, labels,
//     label_len);
//   }
// }

// void FullyConnectedLayer::computeMeanSquaredErrors(
//     uint32_t batch_indx, uint32_t batch_size, const uint32_t* truth_indices,
//     const float* truth_values, uint32_t truth_len) {
//   if (_sparse_dim == _dim) {
//     if (truth_len == _dim) {
//       computeMeanSquaredErrorsImpl<true, true>(
//           batch_indx, batch_size, truth_indices, truth_values, truth_len);
//     } else {
//       computeMeanSquaredErrorsImpl<true, false>(
//           batch_indx, batch_size, truth_indices, truth_values, truth_len);
//     }
//   } else {
//     if (truth_len == _dim) {
//       computeMeanSquaredErrorsImpl<false, true>(
//           batch_indx, batch_size, truth_indices, truth_values, truth_len);
//     } else {
//       computeMeanSquaredErrorsImpl<false, false>(
//           batch_indx, batch_size, truth_indices, truth_values, truth_len);
//     }
//   }
// }

// template <bool DENSE>
// void FullyConnectedLayer::computeSoftmaxErrorsImpl(uint32_t batch_indx,
//                                                    uint32_t batch_size,
//                                                    const uint32_t* labels,
//                                                    uint32_t label_len) {
//   float frac = 1.0 / label_len;

//   for (uint64_t n = 0; n < _active_lens[batch_indx]; n++) {
//     // Because DENSE is known at compile time the compiler can remove this
//     // conditional
//     uint32_t act_neuron = DENSE ? n : _active_neurons[batch_indx][n];
//     if (std::find(labels, labels + label_len, act_neuron) !=
//         labels + label_len) {
//       _errors[batch_indx][n] =
//           (frac - _activations[batch_indx][n]) / batch_size;
//     } else {
//       _errors[batch_indx][n] = -_activations[batch_indx][n] / batch_size;
//     }
//   }
// }

// template <bool DENSE, bool TRUTH_DENSE>
// void FullyConnectedLayer::computeMeanSquaredErrorsImpl(
//     uint32_t batch_indx, uint32_t batch_size, const uint32_t* truth_indices,
//     const float* truth_values, uint32_t truth_len) {
//   for (uint64_t n = 0; n < _active_lens[batch_indx]; n++) {
//     uint32_t act_neuron = DENSE ? n : _active_neurons[batch_indx][n];
//     float matching_truth_value;
//     if (TRUTH_DENSE) {
//       matching_truth_value = truth_values[act_neuron];
//     } else {
//       const unsigned int* itr =
//           std::find(truth_indices, truth_indices + truth_len, act_neuron);
//       if (itr != truth_indices + truth_len) {
//         matching_truth_value = truth_values[std::distance(truth_indices,
//         itr)];
//       } else {
//         matching_truth_value = 0.0;
//       }
//     }
//     _errors[batch_indx][n] =
//         2 * (matching_truth_value - _activations[batch_indx][n]) /
//         batch_size;
//   }
// }

template <bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::selectActiveNeurons(uint32_t batch_indx,
                                              const uint32_t* indices,
                                              const float* values, uint32_t len,
                                              uint32_t* indices_out,
                                              const uint32_t* labels,
                                              uint32_t label_len) {
  if (!DENSE) {
    std::unordered_set<uint32_t> active_set;

    for (uint32_t i = 0; i < label_len; i++) {
      active_set.insert(labels[i]);
    }

    uint32_t* hashes = new uint32_t[_hash_table->numTables()];
    if (PREV_DENSE) {
      _hasher->hashSingleDense(values, len, hashes);
    } else {
      _hasher->hashSingleSparse(indices, values, len, hashes);
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
      _active_neurons[batch_indx][cnt++] = labels[i];
      active_set.erase(labels[i]);
    }
    for (auto x : active_set) {
      if (cnt >= _sparse_dim) {
        break;
      }
      indices_out[cnt++] = x;
    }
  }
}

void FullyConnectedLayer::updateParameters(float lr, uint32_t iter, float B1,
                                           float B2, float eps) {
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