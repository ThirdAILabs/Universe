#include "FullyConnectedLayer.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <random>
#include <unordered_map>

namespace thirdai::bolt {

FullyConnectedLayer::FullyConnectedLayer(
    const FullyConnectedLayerConfig& config, uint64_t prev_dim)
    : _dim(config.dim),
      _prev_dim(prev_dim),
      _sparse_dim(config.sparsity * config.dim),
      _sparsity(config.sparsity),
      _is_shallow(false),
      _shallow_save(false),

      // trainable parameter not present in config file
      // TODO(Shubh) : should we add a trainable parameter to the config file?
      _trainable(true),

      _act_func(config.act_func),
      _weights(config.dim * prev_dim),
      _w_gradient(config.dim * prev_dim, 0),
      _w_momentum(config.dim * prev_dim, 0),
      _w_velocity(config.dim * prev_dim, 0),
      _biases(config.dim),
      _b_gradient(config.dim, 0),
      _b_momentum(config.dim, 0),
      _b_velocity(config.dim, 0),
      _sampling_config(config.sampling_config),
      _prev_is_active(_prev_dim, false),
      _is_active(config.dim, false),
      _sampling_mode(LSHSamplingMode::Default) {
  std::random_device rd;
  std::default_random_engine eng(rd());
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(_weights.begin(), _weights.end(), [&]() { return dist(eng); });
  std::generate(_biases.begin(), _biases.end(), [&]() { return dist(eng); });

  if (_sparsity < 1.0) {
    initSparseDatastructures(rd);
  }
}

void FullyConnectedLayer::forward(const BoltVector& input, BoltVector& output,
                                  const BoltVector* labels) {
  if (output.active_neurons == nullptr) {
    if (input.len == _prev_dim) {
      // TODO(Nicholas): Re-implement this case with dense matrix library
      forwardImpl<true, true>(input, output, labels);
    } else {
      forwardImpl<true, false>(input, output, labels);
    }
  } else {
    if (input.len == _prev_dim) {
      forwardImpl<false, true>(input, output, labels);
    } else {
      forwardImpl<false, false>(input, output, labels);
    }
  }
}

template <bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::forwardImpl(const BoltVector& input,
                                      BoltVector& output,
                                      const BoltVector* labels) {
  assert((input.len < _prev_dim && !PREV_DENSE) ||
         (input.len == _prev_dim && PREV_DENSE));
  assert((input.active_neurons == nullptr && PREV_DENSE) ||
         (input.active_neurons != nullptr && !PREV_DENSE));
  assert((output.len < _dim && !DENSE) || (output.len == _dim && DENSE));
  assert((output.active_neurons == nullptr && DENSE) ||
         (output.active_neurons != nullptr && !DENSE));
  assert(labels == nullptr || labels->len > 0);

  selectActiveNeurons<DENSE, PREV_DENSE>(input, output, labels);

  float max_act = 0;
  uint32_t len_out = DENSE ? _dim : _sparse_dim;
  std::fill_n(output.gradients, len_out, 0);

  _prev_is_dense = PREV_DENSE;
  _this_is_dense = DENSE;

  if (!DENSE && !PREV_DENSE) {
    std::unique_ptr<ActiveNeuronsPair> active_pairs =
        std::make_unique<ActiveNeuronsPair>(std::vector<uint64_t>(),
                                            std::vector<uint64_t>());
    for (uint64_t i = 0; i < input.len; i++) {
      active_pairs->first.push_back(input.active_neurons[i]);
    }
    for (uint64_t n = 0; n < len_out; n++) {
      active_pairs->second.push_back(output.active_neurons[n]);
    }
#pragma omp critical
    _active_pairs.push_back(std::move(active_pairs));
  }

  if (!DENSE) {
    for (uint64_t n = 0; n < len_out; n++) {
      uint64_t act_neuron = output.active_neurons[n];
      _is_active[act_neuron] = true;
    }
  }

  if (!PREV_DENSE) {
    for (uint64_t i = 0; i < input.len; i++) {
      uint64_t act_neuron = input.active_neurons[i];
      _prev_is_active[act_neuron] = true;
    }
  }

  for (uint64_t n = 0; n < len_out; n++) {
    // Because DENSE is known at compile time the compiler can remove this
    // conditional
    uint64_t act_neuron = DENSE ? n : output.active_neurons[n];
    assert(act_neuron < _dim);
    float act = _biases[act_neuron];
    for (uint64_t i = 0; i < input.len; i++) {
      // Because PREV_DENSE is known at compile time the compiler can remove
      // this conditional
      uint32_t prev_act_neuron = PREV_DENSE ? i : input.active_neurons[i];
      assert(prev_act_neuron < _prev_dim);
      act += _weights[act_neuron * _prev_dim + prev_act_neuron] *
             input.activations[i];
    }

    assert(!std::isnan(act));

    switch (_act_func) {
      case ActivationFunction::ReLU:
        if (act < 0) {
          output.activations[n] = 0;
        } else {
          output.activations[n] = act;
        }
        break;
      case ActivationFunction::Softmax:
        output.activations[n] = act;
        if (max_act < act) {
          max_act = act;
        }
        break;
      case ActivationFunction::Sigmoid:
        output.activations[n] = 1 / (1 + std::exp(-act));
        break;
      case ActivationFunction::Linear:
        output.activations[n] = act;
        break;
      case ActivationFunction::Tanh:
        output.activations[n] = static_cast<float>(std::tanh(act));
        break;
    }
  }

  if (_act_func == ActivationFunction::Softmax) {
    float total = 0;
    for (uint64_t n = 0; n < len_out; n++) {
      output.activations[n] = std::exp(output.activations[n] - max_act);
      total += output.activations[n];
    }
    for (uint64_t n = 0; n < len_out; n++) {
      output.activations[n] /= (total + EPS);
      assert(!std::isnan(output.activations[n]));
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

template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::backpropagateImpl(BoltVector& input,
                                            BoltVector& output) {
  assert((input.len < _prev_dim && !PREV_DENSE) ||
         (input.len == _prev_dim && PREV_DENSE));
  assert((input.active_neurons == nullptr && PREV_DENSE) ||
         (input.active_neurons != nullptr && !PREV_DENSE));
  assert((output.len < _dim && !DENSE) || (output.len == _dim && DENSE));
  assert((output.active_neurons == nullptr && DENSE) ||
         (output.active_neurons != nullptr && !DENSE));

  uint32_t len_out = DENSE ? _dim : _sparse_dim;

  for (uint64_t n = 0; n < len_out; n++) {
    assert(!std::isnan(output.gradients[n]));
    output.gradients[n] *= actFuncDerivative(output.activations[n], _act_func);
    assert(!std::isnan(output.gradients[n]));
    // Because DENSE is known at compile time the compiler can remove this
    // conditional
    uint32_t act_neuron = DENSE ? n : output.active_neurons[n];
    assert(act_neuron < _dim);
    for (uint64_t i = 0; i < input.len; i++) {
      // Because PREV_DENSE is known at compile time the compiler can remove
      // this conditional
      uint32_t prev_act_neuron = PREV_DENSE ? i : input.active_neurons[i];
      assert(prev_act_neuron < _prev_dim);
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
                                              const BoltVector* labels) {
  if (DENSE) {
    return;
  }

  std::unordered_set<uint32_t> active_set;

  uint32_t label_len = labels != nullptr ? labels->len : 0;
  for (uint32_t i = 0; i < label_len; i++) {
    assert(labels->active_neurons[i] < _dim);
    active_set.insert(labels->active_neurons[i]);
  }

  std::vector<uint32_t> hashes(_hasher->numTables());
  if (PREV_DENSE) {
    _hasher->hashSingleDense(input.activations, input.len, hashes.data());
  } else {
    _hasher->hashSingleSparse(input.active_neurons, input.activations,
                              input.len, hashes.data());
  }

  if (_sampling_mode == LSHSamplingMode::FreezeHashTablesWithInsertions) {
    /**
     * QueryBySet just returns a set of the elements in the given buckets of the
     * hash table.
     *
     * QueryAndInsertForInference returns the set of elements in the given
     * buckets but will also insert the labels (during training only) for the
     * vector into the buckets the vector maps to if they are not already
     * present in the buckets. The intuition is that during sparse inference
     * this will help force the hash tables to map vectors towards buckets that
     * contain their correct labels. This is specific to the output layer.
     *
     * We call QueryAndInsertForInference if the following conditions are met:
     *   1. We have sparse inference enabled.
     *   2. Activation = Softmax or Sigmoid, meaning it's a classification task,
     *      and that the given layer is the last layer, as this is the only
     *      place where we use these activation functions.
     */
    _hash_table->queryAndInsertForInference(hashes.data(), active_set,
                                            _sparse_dim);
  } else {
    _hash_table->queryBySet(hashes.data(), active_set);
  }
  // making the first value in hashes as random number , because
  // rand() is not thread safe, and this will be random for different inputs but
  // for an input it will be deterministic.
  if (active_set.size() < _sparse_dim) {
    uint32_t rand_offset = (hashes[0]) % _dim;
    while (active_set.size() < _sparse_dim) {
      active_set.insert(_rand_neurons[rand_offset++]);
      rand_offset = rand_offset % _dim;
    }
  }

  uint32_t cnt = 0;
  for (uint32_t i = 0; i < label_len; i++) {
    if (cnt == _sparse_dim) {
      break;
    }
    output.active_neurons[cnt++] = labels->active_neurons[i];
    active_set.erase(labels->active_neurons[i]);
  }

  for (auto x : active_set) {
    if (cnt == _sparse_dim) {
      break;
    }
    assert(x < _dim);
    output.active_neurons[cnt++] = x;
  }
}

void FullyConnectedLayer::updateParameters(float lr, uint32_t iter, float B1,
                                           float B2, float eps) {
  float B1_bias_corrected = static_cast<float>(1 - pow(B1, iter));
  float B2_bias_corrected = static_cast<float>(1 - pow(B2, iter));

  // if the layer is non-trainable, skip updating the parameters
  if (!_trainable) {
    cleanupWithinBatchVars();
    return;
  }

  // continue if trainable layer
  if (!_prev_is_dense && !_this_is_dense) {
    updateSparseSparseWeightParameters(lr, B1, B2, eps, B1_bias_corrected,
                                       B2_bias_corrected);
  } else if (!_prev_is_dense && _this_is_dense) {
    updateSparseDenseWeightParameters(lr, B1, B2, eps, B1_bias_corrected,
                                      B2_bias_corrected);
  } else if (_prev_is_dense && !_this_is_dense) {
    updateDenseSparseWeightParameters(lr, B1, B2, eps, B1_bias_corrected,
                                      B2_bias_corrected);
  } else {
    updateDenseDenseWeightParameters(lr, B1, B2, eps, B1_bias_corrected,
                                     B2_bias_corrected);
  }

  updateBiasParameters(lr, B1, B2, eps, B1_bias_corrected, B2_bias_corrected);

  cleanupWithinBatchVars();
}

inline void FullyConnectedLayer::updateSparseSparseWeightParameters(
    float lr, float B1, float B2, float eps, float B1_bias_corrected,
    float B2_bias_corrected) {
#pragma omp parallel for default(none) \
    shared(lr, B1, B1_bias_corrected, B2, B2_bias_corrected, eps)
  // TODO(Josh): It is possible to update the same active pair multiple times
  // with the full gradient. Possible solutions include:
  // 1. Including the gradient in the active pair and doing an update multiple
  //    times with the smaller gradients. This is effectively equal to batch
  //     size 1, but we basically already have batch size 1 with sparse sparse.
  // 2. Having a bloom filter where we cheaply hash the active pairs to a bloom
  //    filter bit, and if it is already set in the bloom filter skip it,
  //    otherwise set the bit and do the gradient update.
  for (uint32_t pair_id = 0; pair_id < _active_pairs.size();  // NOLINT
       pair_id++) {
    // MSVC doesn't like if we iterate over objects, only integers
    // (but clang-tidy wants the range based for loop, so we need NOLINT above)
    const auto& active_pair = _active_pairs[pair_id];
    for (uint64_t prev_neuron : active_pair->first) {
      for (uint64_t cur_neuron : active_pair->second) {
        updateSingleWeightParameters(prev_neuron, cur_neuron, lr, B1, B2, eps,
                                     B1_bias_corrected, B2_bias_corrected);
      }
    }
  }
}

inline void FullyConnectedLayer::updateSparseDenseWeightParameters(
    float lr, float B1, float B2, float eps, float B1_bias_corrected,
    float B2_bias_corrected) {
  // TODO(josh): Possibly reorder these loops to put the _is_active on the
  // outside? I worry this will hurt cache efficiency on the gradient lookups.
  // It also might eventually depend on the underlying memory layout of the
  // weights/parameters, which will be optimized for easy vectorization.
#pragma omp parallel for default(none) \
    shared(lr, B1, B1_bias_corrected, B2, B2_bias_corrected, eps)
  for (uint64_t cur_neuron = 0; cur_neuron < _dim; cur_neuron++) {
    for (uint64_t prev_neuron = 0; prev_neuron < _prev_dim; prev_neuron++) {
      if (_prev_is_active[prev_neuron]) {
        updateSingleWeightParameters(prev_neuron, cur_neuron, lr, B1, B2, eps,
                                     B1_bias_corrected, B2_bias_corrected);
      }
    }
  }
}

inline void FullyConnectedLayer::updateDenseSparseWeightParameters(
    float lr, float B1, float B2, float eps, float B1_bias_corrected,
    float B2_bias_corrected) {
#pragma omp parallel for default(none) \
    shared(lr, B1, B1_bias_corrected, B2, B2_bias_corrected, eps)
  for (uint64_t cur_neuron = 0; cur_neuron < _dim; cur_neuron++) {
    if (_is_active[cur_neuron]) {
      for (uint64_t prev_neuron = 0; prev_neuron < _prev_dim; prev_neuron++) {
        updateSingleWeightParameters(prev_neuron, cur_neuron, lr, B1, B2, eps,
                                     B1_bias_corrected, B2_bias_corrected);
      }
    }
  }
}

inline void FullyConnectedLayer::updateDenseDenseWeightParameters(
    float lr, float B1, float B2, float eps, float B1_bias_corrected,
    float B2_bias_corrected) {
#pragma omp parallel for default(none) \
    shared(lr, B1, B1_bias_corrected, B2, B2_bias_corrected, eps)
  for (uint64_t cur_neuron = 0; cur_neuron < _dim; cur_neuron++) {
    for (uint64_t prev_neuron = 0; prev_neuron < _prev_dim; prev_neuron++) {
      updateSingleWeightParameters(prev_neuron, cur_neuron, lr, B1, B2, eps,
                                   B1_bias_corrected, B2_bias_corrected);
    }
  }
}

inline void FullyConnectedLayer::updateBiasParameters(float lr, float B1,
                                                      float B2, float eps,
                                                      float B1_bias_corrected,
                                                      float B2_bias_corrected) {
#pragma omp parallel for default(none) \
    shared(lr, B1, B1_bias_corrected, B2, B2_bias_corrected, eps)
  for (uint64_t cur_neuron = 0; cur_neuron < _dim; cur_neuron++) {
    if (!_this_is_dense && !_is_active[cur_neuron]) {
      continue;
    }

    float grad = _b_gradient[cur_neuron];
    assert(!std::isnan(grad));

    _b_momentum[cur_neuron] = B1 * _b_momentum[cur_neuron] + (1 - B1) * grad;
    _b_velocity[cur_neuron] =
        B2 * _b_velocity[cur_neuron] + (1 - B2) * grad * grad;

    assert(!std::isnan(_b_momentum[cur_neuron]));
    assert(!std::isnan(_b_velocity[cur_neuron]));

    _biases[cur_neuron] +=
        lr * (_b_momentum[cur_neuron] / B1_bias_corrected) /
        (std::sqrt(_b_velocity[cur_neuron] / B2_bias_corrected) + eps);
    assert(!std::isnan(_biases[cur_neuron]));

    _b_gradient[cur_neuron] = 0;
    _is_active[cur_neuron] = false;
  }
}

inline void FullyConnectedLayer::cleanupWithinBatchVars() {
  _active_pairs.clear();
  for (uint64_t i = 0; i < _prev_dim; i++) {
    _prev_is_active[i] = false;
  }
  for (uint64_t n = 0; n < _dim; n++) {
    _is_active[n] = false;
  }
}

inline void FullyConnectedLayer::updateSingleWeightParameters(
    uint64_t prev_neuron, uint64_t cur_neuron, float lr, float B1, float B2,
    float eps, float B1_bias_corrected, float B2_bias_corrected) {
  auto indx = cur_neuron * _prev_dim + prev_neuron;
  float grad = _w_gradient[indx];
  assert(!std::isnan(grad));

  _w_momentum[indx] = B1 * _w_momentum[indx] + (1 - B1) * grad;
  _w_velocity[indx] = B2 * _w_velocity[indx] + (1 - B2) * grad * grad;
  assert(!std::isnan(_w_momentum[indx]));
  assert(!std::isnan(_w_velocity[indx]));

  _weights[indx] += lr * (_w_momentum[indx] / B1_bias_corrected) /
                    (std::sqrt(_w_velocity[indx] / B2_bias_corrected) + eps);
  assert(!std::isnan(_weights[indx]));

  _w_gradient[indx] = 0;
}

inline void FullyConnectedLayer::initSparseDatastructures(
    std::random_device& rd) {
  _hasher = assignHashFunction(_sampling_config, _prev_dim);

  _hash_table = std::make_unique<hashtable::SampledHashTable<uint32_t>>(
      _sampling_config.num_tables, _sampling_config.reservoir_size,
      1 << _sampling_config.range_pow);

  /* Initializing hence, we need to force build the hash tables
   * Hence, force_build is true here in buildHashTablesImpl(force_build)
   */
  buildHashTablesImpl(true);

  _rand_neurons = std::vector<uint32_t>(_dim);

  std::iota(_rand_neurons.begin(), _rand_neurons.end(), 0);
  std::shuffle(_rand_neurons.begin(), _rand_neurons.end(), rd);
}

inline void FullyConnectedLayer::deinitSparseDatastructures() {
  _hasher = {};
  _hash_table = {};
  _rand_neurons = {};
}

void FullyConnectedLayer::buildHashTablesImpl(bool force_build) {
  if ((!_trainable && !force_build) || _sparsity >= 1.0 || hashTablesFrozen()) {
    return;
  }
  uint64_t num_tables = _hash_table->numTables();
  // TODO(nicholas): hashes could be array with size max(batch size, dim) that
  // is allocated once
  std::vector<uint32_t> hashes(num_tables * _dim);
#pragma omp parallel for default(none) shared(num_tables, hashes)
  for (uint64_t n = 0; n < _dim; n++) {
    _hasher->hashSingleDense(_weights.data() + n * _prev_dim, _prev_dim,
                             hashes.data() + n * num_tables);
  }

  _hash_table->clearTables();
  _hash_table->insertSequential(_dim, 0, hashes.data());
}

/* setting force_build to false. force_build true only when setting weights or
 * initializing
 */
void FullyConnectedLayer::buildHashTables() { buildHashTablesImpl(false); }

void FullyConnectedLayer::reBuildHashFunction() {
  if (!_trainable || _sparsity >= 1.0 || hashTablesFrozen()) {
    return;
  }
  _hasher = assignHashFunction(_sampling_config, _prev_dim);
}

float* FullyConnectedLayer::getWeights() const {
  float* weights_copy = new float[_dim * _prev_dim];
  std::copy(_weights.begin(), _weights.end(), weights_copy);

  return weights_copy;
}

void FullyConnectedLayer::setTrainable(bool trainable) {
  _trainable = trainable;
}

bool FullyConnectedLayer::getTrainable() const { return _trainable; }

float* FullyConnectedLayer::getBiases() const {
  float* biases_copy = new float[_dim];
  std::copy(_biases.begin(), _biases.end(), biases_copy);

  return biases_copy;
}

void FullyConnectedLayer::setWeights(const float* new_weights) {
  std::copy(new_weights, new_weights + _dim * _prev_dim, _weights.begin());

  /* Setting weights => we need to force build the hash tables
   * Hence, force_build is true here in buildHashTablesImpl(force_build)
   */
  buildHashTablesImpl(true);
}

void FullyConnectedLayer::setBiases(const float* new_biases) {
  std::copy(new_biases, new_biases + _dim, _biases.begin());
}

void FullyConnectedLayer::setShallow(bool shallow) {
  /**
   * Initialize optimizer only when layer is currently shallow and shallow is
   * false. Remove optimizer only if the layer is currently non-shallow but
   * shallow is true
   */
  if (!_is_shallow && shallow) {
    this->removeOptimizer();
  } else if (_is_shallow && !shallow) {
    this->initOptimizer();
  }
  _is_shallow = shallow;
}

void FullyConnectedLayer::setShallowSave(bool shallow) {
  _shallow_save = shallow;
}

void FullyConnectedLayer::setSparsity(float sparsity) {
  deinitSparseDatastructures();
  _sparsity = sparsity;
  // TODO(josh): Right now this is using the autotuning for DWTA even if this
  // hash function isn't DWTA. Add autotuning for other hash function types.
  _sampling_config =
      FullyConnectedLayerConfig(_dim, _sparsity, _act_func).sampling_config;
  _sparse_dim = _sparsity * _dim;
  std::random_device rd;
  initSparseDatastructures(rd);
}

void FullyConnectedLayer::initOptimizer() {
  _w_gradient.assign(_dim * _prev_dim, 0);
  _w_momentum.assign(_dim * _prev_dim, 0);
  _w_velocity.assign(_dim * _prev_dim, 0);

  _b_gradient.assign(_dim, 0);
  _b_momentum.assign(_dim, 0);
  _b_velocity.assign(_dim, 0);
}

void FullyConnectedLayer::removeOptimizer() {
  _w_gradient.clear();
  _w_momentum.clear();
  _w_velocity.clear();

  _b_gradient.clear();
  _b_momentum.clear();
  _b_velocity.clear();
}

void FullyConnectedLayer::buildLayerSummary(std::stringstream& summary,
                                            bool detailed) const {
  summary << "dim=" << _dim << ", sparsity=" << _sparsity << ", act_func=";
  switch (_act_func) {
    case ActivationFunction::ReLU:
      summary << "ReLU";
      break;
    case ActivationFunction::Softmax:
      summary << "Softmax";
      break;
    case ActivationFunction::Sigmoid:
      summary << "Sigmoid";
      break;
    case ActivationFunction::Linear:
      summary << "Linear";
      break;
    case ActivationFunction::Tanh:
      summary << "Tanh";
      break;
  }

  if (!detailed) {
    summary << "\n";
    return;
  }

  summary << " (hashes_per_table=" << _sampling_config.hashes_per_table
          << ", num_tables=" << _sampling_config.num_tables
          << ", range_pow=" << _sampling_config.range_pow
          << ", resevoir_size=" << _sampling_config.reservoir_size
          << ", hash_function="
          << getHashString(_sampling_config._hash_function) << ")";

  summary << "\n";
}

}  // namespace thirdai::bolt