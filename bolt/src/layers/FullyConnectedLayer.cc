#include "FullyConnectedLayer.h"
#include <wrappers/src/EigenDenseWrapper.h>
#include <bolt/src/layers/LayerUtils.h>
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/util/Constants.h>
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
    const FullyConnectedLayerConfig& config, uint64_t prev_dim,
    bool disable_sparse_parameter_updates)
    : _dim(config.getDim()),
      _prev_dim(prev_dim),
      _sparse_dim(config.getSparsity() * config.getDim()),
      _sparsity(config.getSparsity()),

      // trainable parameter not present in config file
      // TODO(Shubh) : should we add a trainable parameter to the config file?
      _trainable(true),
      _hash_tables_frozen(false),
      _act_func(config.getActFunc()),
      _weights(config.getDim() * prev_dim),
      _biases(config.getDim()),
      _disable_sparse_parameter_updates(disable_sparse_parameter_updates),
      _sampling_mode(BoltSamplingMode::LSH),
      _prev_is_active(prev_dim, false),
      _is_active(config.getDim(), false) {
  std::random_device rd;
  std::default_random_engine eng(rd());
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(_weights.begin(), _weights.end(), [&]() { return dist(eng); });
  std::generate(_biases.begin(), _biases.end(), [&]() { return dist(eng); });

  if (_sparsity < 1.0) {
    initSamplingDatastructures(config.getSamplingConfig(), rd);
  }

  initOptimizer();

  initActiveNeuronsTrackers();
}

void FullyConnectedLayer::forward(const BoltVector& input, BoltVector& output,
                                  const BoltVector* labels) {
  if (output.isDense()) {
    if (input.isDense()) {
      eigenDenseDenseForward(input, output);
    } else {
      forwardImpl</*DENSE=*/true, /*PREV_DENSE=*/false>(input, output, labels);
    }
  } else {
    if (input.isDense()) {
      forwardImpl</*DENSE=*/false, /*PREV_DENSE=*/true>(input, output, labels);
    } else {
      forwardImpl</*DENSE=*/false, /*PREV_DENSE=*/false>(input, output, labels);
    }
  }
}

template <bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::forwardImpl(const BoltVector& input,
                                      BoltVector& output,
                                      const BoltVector* labels) {
  assert((input.len <= _prev_dim && !PREV_DENSE) ||
         (input.len == _prev_dim && PREV_DENSE));
  assert((input.active_neurons == nullptr && PREV_DENSE) ||
         (input.active_neurons != nullptr && !PREV_DENSE));
  assert((output.len <= _dim && !DENSE) || (output.len == _dim && DENSE));
  assert((output.active_neurons == nullptr && DENSE) ||
         (output.active_neurons != nullptr && !DENSE));
  assert(labels == nullptr || labels->len > 0);

  selectActiveNeurons<DENSE, PREV_DENSE>(input, output, labels);

  float max_act = 0;
  uint32_t len_out = nonzerosInOutput<DENSE>();

  // active neurons are for updateParameters so we only mark them if trainable
  if (_trainable) {
    // TODO(david) this is not needed for inference, we can optionally remove
    // this with some refactoring if we want slightly faster inference
    markActiveNeuronsForUpdate<DENSE, PREV_DENSE>(input, output, len_out);
  }

  for (uint64_t n = 0; n < len_out; n++) {
    // Because DENSE is known at compile time the compiler can remove this
    // conditional
    uint64_t act_neuron = output.activeNeuronAtIndex<DENSE>(n);
    assert(act_neuron < _dim);

    float act = _biases[act_neuron];
    for (uint64_t i = 0; i < input.len; i++) {
      // Because PREV_DENSE is known at compile time the compiler can remove
      // this conditional
      uint32_t prev_act_neuron = input.activeNeuronAtIndex<PREV_DENSE>(i);
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

template <bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::markActiveNeuronsForUpdate(const BoltVector& input,
                                                     const BoltVector& output,
                                                     uint32_t len_out) {
  _prev_is_dense = PREV_DENSE;
  _this_is_dense = DENSE;

  if constexpr (!DENSE) {
    for (uint64_t n = 0; n < len_out; n++) {
      uint64_t act_neuron = output.active_neurons[n];
      _is_active[act_neuron] = true;
    }
  }

  if constexpr (!PREV_DENSE) {
    for (uint64_t i = 0; i < input.len; i++) {
      uint64_t act_neuron = input.active_neurons[i];
      _prev_is_active[act_neuron] = true;
    }
  }
}

static void eigenSoftmax(Eigen::Map<Eigen::VectorXf>& outputs) {
  float max_act = outputs.maxCoeff();
  outputs = (outputs.array() - max_act).exp();
  float sum = outputs.sum() + EPS;
  outputs.array() /= sum;
}

void FullyConnectedLayer::eigenDenseDenseForward(const BoltVector& input,
                                                 BoltVector& output) {
  _prev_is_dense = true;
  _this_is_dense = true;

  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      eigen_weights(_weights.data(), _dim, _prev_dim);
  Eigen::Map<Eigen::VectorXf> eigen_biases(_biases.data(), _dim);

  Eigen::Map<Eigen::VectorXf> eigen_input(input.activations, input.len);
  Eigen::Map<Eigen::VectorXf> eigen_output(output.activations, output.len);

  eigen_output.noalias() = eigen_weights * eigen_input;

  eigen_biases.array().addTo(eigen_output);

  switch (_act_func) {
    case ActivationFunction::ReLU:
      eigen_output = eigen_output.array().max(0.0);
      break;
    case ActivationFunction::Softmax:
      eigenSoftmax(eigen_output);
      break;
    case ActivationFunction::Linear:
      break;
    case ActivationFunction::Sigmoid:
      eigen_output = 1 + (-eigen_output.array()).exp();
      eigen_output = eigen_output.array().rsqrt();
      break;
    case ActivationFunction::Tanh:
      eigen_output = eigen_output.array().tanh();
      break;
  }
}

void FullyConnectedLayer::backpropagate(BoltVector& input, BoltVector& output) {
  if (output.isDense()) {
    if (input.isDense()) {
      // This eigen dense dense optimized version seems to give speedup in
      // certain cases but not all, so it is here as an experimental feature
      // that can be enabled when desired.
#if THIRDAI_USE_EIGEN_FOR_BACKPROPAGATE
      eigenDenseDenseBackpropagate<false>(input, output);
#else
      backpropagateImpl<false, true, true>(input, output);
#endif
    } else {
      backpropagateImpl<false, true, false>(input, output);
    }
  } else {
    if (input.isDense()) {
      backpropagateImpl<false, false, true>(input, output);
    } else {
      backpropagateImpl<false, false, false>(input, output);
    }
  }
}

void FullyConnectedLayer::backpropagateInputLayer(BoltVector& input,
                                                  BoltVector& output) {
  if (output.isDense()) {
    if (input.isDense()) {
      // This eigen dense dense optimized version seems to give speedup in
      // certain cases but not all, so it is here as an experimental feature
      // that can be enabled when desired.
#if THIRDAI_USE_EIGEN_FOR_BACKPROP
      eigenDenseDenseBackpropagate<true>(input, output);
#else
      backpropagateImpl</*IS_INPUT=*/true, /*DENSE=*/true, /*PREV_DENSE=*/true>(
          input, output);
#endif
    } else {
      backpropagateImpl</*IS_INPUT=*/true, /*DENSE=*/true,
                        /*PREV_DENSE=*/false>(input, output);
    }
  } else {
    if (input.isDense()) {
      backpropagateImpl</*IS_INPUT=*/true, /*DENSE=*/false,
                        /*PREV_DENSE=*/true>(input, output);
    } else {
      backpropagateImpl</*IS_INPUT=*/true, /*DENSE=*/false,
                        /*PREV_DENSE=*/false>(input, output);
    }
  }
}

template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::backpropagateImpl(BoltVector& input,
                                            BoltVector& output) {
  assert((input.len <= _prev_dim && !PREV_DENSE) ||
         (input.len == _prev_dim && PREV_DENSE));
  assert((input.active_neurons == nullptr && PREV_DENSE) ||
         (input.active_neurons != nullptr && !PREV_DENSE));
  assert((output.len <= _dim && !DENSE) || (output.len == _dim && DENSE));
  assert((output.active_neurons == nullptr && DENSE) ||
         (output.active_neurons != nullptr && !DENSE));
  assert(_weight_optimizer.has_value() && _bias_optimizer.has_value());

  uint32_t len_out = nonzerosInOutput<DENSE>();

  for (uint64_t n = 0; n < len_out; n++) {
    assert(!std::isnan(output.gradients[n]));
    output.gradients[n] *= actFuncDerivative(output.activations[n], _act_func);

    if (output.gradients[n] == 0.0) {
      // Neurons with gradients of 0 will not propagate gradients to weights or
      // the previous layer. We will also likely have a number of 0 gradients
      // with ReLU.
      continue;
    }

    assert(!std::isnan(output.gradients[n]));
    // Because DENSE is known at compile time the compiler can remove this
    // conditional
    uint32_t act_neuron = output.activeNeuronAtIndex<DENSE>(n);
    assert(act_neuron < _dim);

    for (uint64_t i = 0; i < input.len; i++) {
      // Because PREV_DENSE is known at compile time the compiler can remove
      // this conditional
      uint32_t prev_act_neuron = input.activeNeuronAtIndex<PREV_DENSE>(i);
      assert(prev_act_neuron < _prev_dim);

      _weight_optimizer->gradients[act_neuron * _prev_dim + prev_act_neuron] +=
          output.gradients[n] * input.activations[i];
      if constexpr (!FIRST_LAYER) {
        input.gradients[i] +=
            output.gradients[n] *
            _weights[act_neuron * _prev_dim + prev_act_neuron];
      }
    }
    _bias_optimizer->gradients[act_neuron] += output.gradients[n];
  }
}

template <bool FIRST_LAYER>
void FullyConnectedLayer::eigenDenseDenseBackpropagate(BoltVector& input,
                                                       BoltVector& output) {
  assert(_weight_optimizer.has_value() && _bias_optimizer.has_value());

  for (uint32_t n = 0; n < output.len; n++) {
    output.gradients[n] *= actFuncDerivative(output.activations[n], _act_func);
  }

  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      eigen_weights(_weights.data(), _dim, _prev_dim);
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      eigen_weight_grad(_weight_optimizer->gradients.data(), _dim, _prev_dim);

  Eigen::Map<Eigen::VectorXf> eigen_bias_grad(_bias_optimizer->gradients.data(),
                                              _dim);

  Eigen::Map<Eigen::VectorXf> eigen_input(input.activations, input.len);
  Eigen::Map<Eigen::VectorXf> eigen_output_grad(output.gradients, output.len);

  eigen_weight_grad += eigen_output_grad * eigen_input.transpose();
  eigen_output_grad.array().addTo(eigen_bias_grad);

  if constexpr (!FIRST_LAYER) {
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_weights(_weights.data(), _dim, _prev_dim);

    Eigen::Map<Eigen::VectorXf> eigen_input_grad(input.gradients, input.len);

    eigen_input_grad.noalias() = eigen_output_grad.transpose() * eigen_weights;
  }
}

template <bool DENSE, bool PREV_DENSE>
void FullyConnectedLayer::selectActiveNeurons(const BoltVector& input,
                                              BoltVector& output,
                                              const BoltVector* labels) {
  if constexpr (DENSE) {
    return;
  }

  if (useRandomSampling()) {
    randomNeuronSampling(input, output, labels);
  } else {
    lshNeuronSampling<PREV_DENSE>(input, output, labels);
  }
}

static void wrapAroundCopy(const uint32_t* const src, uint64_t src_len,
                           uint32_t* const dest, uint64_t copy_size,
                           uint64_t starting_offset) {
  assert(starting_offset < src_len);
  assert(copy_size <= src_len);

  uint64_t length_to_end = std::min(src_len - starting_offset, copy_size);
  uint64_t length_of_remainder =
      length_to_end < copy_size ? copy_size - length_to_end : 0;

  std::copy(src + starting_offset, src + starting_offset + length_to_end, dest);
  std::copy(src, src + length_of_remainder, dest + length_to_end);
}

void FullyConnectedLayer::randomNeuronSampling(const BoltVector& input,
                                               const BoltVector& output,
                                               const BoltVector* labels) {
  uint32_t label_len = 0;
  if (labels) {
    label_len = std::min<uint64_t>(labels->len, _sparse_dim);
    std::copy(labels->active_neurons, labels->active_neurons + label_len,
              output.active_neurons);
  }

  // This is because rand() is not threadsafe and because we want to make the
  // output more deterministic.
  uint64_t random_offset =
      hashing::simpleIntegerHash(
          // Hack to intepret the float as an integer without doing a
          // conversion.
          *reinterpret_cast<uint32_t*>(&input.activations[0])) %
      _dim;

  uint64_t neurons_to_sample = _sparse_dim - label_len;

  wrapAroundCopy(/* src= */ _rand_neurons.data(), /* src_len= */ _dim,
                 /* dest= */ output.active_neurons + label_len,
                 /* copy_size= */ neurons_to_sample,
                 /* starting_offset= */ random_offset);
}

template <bool PREV_DENSE>
void FullyConnectedLayer::lshNeuronSampling(const BoltVector& input,
                                            BoltVector& output,
                                            const BoltVector* labels) {
  std::unordered_set<uint32_t> active_set;

  uint32_t label_len = labels != nullptr ? labels->len : 0;
  for (uint32_t i = 0; i < label_len; i++) {
    assert(labels->active_neurons[i] < _dim);
    active_set.insert(labels->active_neurons[i]);
  }

  std::vector<uint32_t> hashes(_hasher->numTables());
  if constexpr (PREV_DENSE) {
    _hasher->hashSingleDense(input.activations, input.len, hashes.data());
  } else {
    _hasher->hashSingleSparse(input.active_neurons, input.activations,
                              input.len, hashes.data());
  }

  std::cout << "The hashes are:" << std::endl;
  for (auto x : hashes) {
    std::cout << x << "\t";
  }
  std::cout << std::endl;

  std::cout << "The hashes are:" << std::endl;
  for (auto x : hashes) {
    std::cout << x << "\t";
  }
  std::cout << std::endl;

  switch (_sampling_mode) {
    case BoltSamplingMode::FrequencyRerankingWithInsertions:
      _hash_table->queryWithFrequencyRanking(
          hashes.data(), active_set, _sparse_dim, /*insert_labels=*/true);
      break;
    case BoltSamplingMode::FrequencyReranking:
      _hash_table->queryWithFrequencyRanking(
          hashes.data(), active_set, _sparse_dim, /*insert_labels=*/false);
      break;
    case BoltSamplingMode::FreezeHashTablesWithInsertions:
      _hash_table->queryAndInsertForInference(hashes.data(), active_set,
                                              _sparse_dim);
      break;
    default:
      _hash_table->queryBySet(hashes.data(), active_set);
  }

  std::cout
      << "We have sampled the active neurons and printing the active set from "
         "enquiring the hash tables. The number of active neurons is: "
      << active_set.size() << std::endl;
  for (auto x : active_set) {
    std::cout << x << "\t";
  }
  std::cout << std::endl << "Finished printing the active set" << std::endl;

  if (active_set.size() < _sparse_dim) {
    // here we use hashes[0] as our random number because rand() is not thread
    // safe and we want to have deterministic sampling.
    uint32_t rand_offset = hashes.at(0) % _dim;
    while (active_set.size() < _sparse_dim) {
      active_set.insert(_rand_neurons[rand_offset++]);
      rand_offset = rand_offset % _dim;
    }
  }

  std::cout << "We have added more neurons to the active neurons set. The "
               "number of active neurons is: "
            << active_set.size() << std::endl;
  for (auto x : active_set) {
    std::cout << x << "\t";
  }
  std::cout << std::endl << "Finished printing the active set" << std::endl;

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

  if (!_trainable) {
    cleanupWithinBatchVars();
    return;
  }

  /*
   * In distributed setting, as of now the updates are dense as we
   * are averaging the gradient over multiple training examples.
   *
   * //NOLINT because, clang was producing error as same function is
   * being called in two different if-else blocks. However, the content
   * inside the _is_distributed block might change with time. Hence,
   * was thinking of having different blocks. It also make is visually
   * more clear.
   */
  if (_disable_sparse_parameter_updates ||
      (_prev_is_dense && _this_is_dense)) {  // NOLINT
    updateDenseDenseWeightParameters(lr, B1, B2, eps, B1_bias_corrected,
                                     B2_bias_corrected);
  } else if (!_prev_is_dense && !_this_is_dense) {
    updateSparseSparseWeightParameters(lr, B1, B2, eps, B1_bias_corrected,
                                       B2_bias_corrected);
  } else if (!_prev_is_dense && _this_is_dense) {
    updateSparseDenseWeightParameters(lr, B1, B2, eps, B1_bias_corrected,
                                      B2_bias_corrected);
  } else if (_prev_is_dense && !_this_is_dense) {
    updateDenseSparseWeightParameters(lr, B1, B2, eps, B1_bias_corrected,
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
  for (uint64_t cur_neuron = 0; cur_neuron < _dim; cur_neuron++) {
    if (!_is_active[cur_neuron]) {
      continue;
    }
    for (uint64_t prev_neuron = 0; prev_neuron < _prev_dim; prev_neuron++) {
      if (_prev_is_active[prev_neuron]) {
        updateSingleWeightParameters(prev_neuron, cur_neuron, lr, B1, B2, eps,
                                     B1_bias_corrected, B2_bias_corrected);
      }
    }
  }
}

inline void FullyConnectedLayer::updateSparseDenseWeightParameters(
    float lr, float B1, float B2, float eps, float B1_bias_corrected,
    float B2_bias_corrected) {
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
    if (!_is_active[cur_neuron]) {
      continue;
    }
    for (uint64_t prev_neuron = 0; prev_neuron < _prev_dim; prev_neuron++) {
      updateSingleWeightParameters(prev_neuron, cur_neuron, lr, B1, B2, eps,
                                   B1_bias_corrected, B2_bias_corrected);
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
  assert(_bias_optimizer.has_value());
#pragma omp parallel for default(none) \
    shared(lr, B1, B1_bias_corrected, B2, B2_bias_corrected, eps)
  for (uint64_t cur_neuron = 0; cur_neuron < _dim; cur_neuron++) {
    if (!_disable_sparse_parameter_updates && !_this_is_dense &&
        !_is_active[cur_neuron]) {
      continue;
    }

    float grad = _bias_optimizer->gradients[cur_neuron];
    assert(!std::isnan(grad));

    _bias_optimizer->momentum[cur_neuron] =
        B1 * _bias_optimizer->momentum[cur_neuron] + (1 - B1) * grad;
    _bias_optimizer->velocity[cur_neuron] =
        B2 * _bias_optimizer->velocity[cur_neuron] + (1 - B2) * grad * grad;

    assert(!std::isnan(_bias_optimizer->momentum[cur_neuron]));
    assert(!std::isnan(_bias_optimizer->velocity[cur_neuron]));

    _biases[cur_neuron] +=
        lr * (_bias_optimizer->momentum[cur_neuron] / B1_bias_corrected) /
        (std::sqrt(_bias_optimizer->velocity[cur_neuron] / B2_bias_corrected) +
         eps);
    assert(!std::isnan(_biases[cur_neuron]));

    _bias_optimizer->gradients[cur_neuron] = 0;
  }
}

inline void FullyConnectedLayer::cleanupWithinBatchVars() {
  std::fill(_prev_is_active.begin(), _prev_is_active.end(), 0);
  std::fill(_is_active.begin(), _is_active.end(), 0);
}

inline void FullyConnectedLayer::updateSingleWeightParameters(
    uint64_t prev_neuron, uint64_t cur_neuron, float lr, float B1, float B2,
    float eps, float B1_bias_corrected, float B2_bias_corrected) {
  assert(_weight_optimizer.has_value());

  auto indx = cur_neuron * _prev_dim + prev_neuron;
  float grad = _weight_optimizer->gradients[indx];
  assert(!std::isnan(grad));

  _weight_optimizer->momentum[indx] =
      B1 * _weight_optimizer->momentum[indx] + (1 - B1) * grad;
  _weight_optimizer->velocity[indx] =
      B2 * _weight_optimizer->velocity[indx] + (1 - B2) * grad * grad;
  assert(!std::isnan(_weight_optimizer->momentum[indx]));
  assert(!std::isnan(_weight_optimizer->velocity[indx]));

  _weights[indx] +=
      lr * (_weight_optimizer->momentum[indx] / B1_bias_corrected) /
      (std::sqrt(_weight_optimizer->velocity[indx] / B2_bias_corrected) + eps);
  assert(!std::isnan(_weights[indx]));

  _weight_optimizer->gradients[indx] = 0;
}

void FullyConnectedLayer::initSamplingDatastructures(
    const SamplingConfigPtr& sampling_config, std::random_device& rd) {
  if (sampling_config->isRandomSampling()) {
    _sampling_mode = BoltSamplingMode::RandomSampling;
  } else {
    _sampling_mode = BoltSamplingMode::LSH;
  }

  _hasher = sampling_config->getHashFunction(_prev_dim);

  _hash_table = sampling_config->getHashTable();

  /* Initializing hence, we need to force build the hash tables
   * Hence, force_build is true here in buildHashTablesImpl(force_build)
   */
  buildHashTablesImpl(true);

  _rand_neurons = std::vector<uint32_t>(_dim);

  std::iota(_rand_neurons.begin(), _rand_neurons.end(), 0);
  std::shuffle(_rand_neurons.begin(), _rand_neurons.end(), rd);
}

inline void FullyConnectedLayer::deinitSamplingDatastructures() {
  _hasher = {};
  _hash_table = {};
  _rand_neurons = {};
}

void FullyConnectedLayer::buildHashTablesImpl(bool force_build) {
  if ((!_trainable && !force_build) || _sparsity >= 1.0 || hashTablesFrozen() ||
      useRandomSampling()) {
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
  if (!_trainable || _sparsity >= 1.0 || hashTablesFrozen() ||
      useRandomSampling()) {
    return;
  }

  _hasher = _hasher->copyWithNewSeeds();
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

void FullyConnectedLayer::setWeightGradients(
    const float* update_weight_gradient) {
  assert(_weight_optimizer.has_value());

  std::copy(update_weight_gradient, update_weight_gradient + _dim * _prev_dim,
            _weight_optimizer->gradients.begin());
}

void FullyConnectedLayer::setBiasesGradients(
    const float* update_bias_gradient) {
  assert(_bias_optimizer.has_value());
  std::copy(update_bias_gradient, update_bias_gradient + _dim,
            _bias_optimizer->gradients.begin());
}

float* FullyConnectedLayer::getBiasesGradient() {
  assert(_bias_optimizer.has_value());
  return _bias_optimizer->gradients.data();
}

float* FullyConnectedLayer::getWeightsGradient() {
  assert(_weight_optimizer.has_value());

  return _weight_optimizer->gradients.data();
}

std::vector<float> FullyConnectedLayer::getWeightsByNeuron(uint32_t neuron_id) {
  if (neuron_id >= _dim) {
    throw std::invalid_argument(
        "Passed in neuron_id too large for this layer. Should be less than the "
        "output dim of " +
        std::to_string(_dim) + ".");
  }

  std::vector<float> embedding(_weights.begin() + neuron_id * _prev_dim,
                               _weights.begin() + (neuron_id + 1) * _prev_dim);
  return embedding;
}

void FullyConnectedLayer::setSparsity(float sparsity) {
  // deinitSamplingDatastructures();
  _sparsity = sparsity;

  _sparse_dim = _sparsity * _dim;

  // TODO(josh): Right now this is using the autotuning for DWTA even if this
  // hash function isn't DWTA. Add autotuning for other hash function types.
  // if (_sparsity < 1.0) {
  //   auto sampling_config = DWTASamplingConfig::autotune(_dim, _sparsity);
  //   std::random_device rd;
  //   initSamplingDatastructures(sampling_config, rd);
  // }
}

void FullyConnectedLayer::initOptimizer() {
  if (!_weight_optimizer || !_bias_optimizer) {
    _weight_optimizer = AdamOptimizer(_dim * _prev_dim);
    _bias_optimizer = AdamOptimizer(_dim);
  }
}

void FullyConnectedLayer::initActiveNeuronsTrackers() {
  _prev_is_active.assign(_prev_dim, false);
  _is_active.assign(_dim, false);
}

void FullyConnectedLayer::buildLayerSummary(std::stringstream& summary,
                                            bool detailed) const {
  summary << "dim=" << _dim << ", sparsity=" << _sparsity << ", act_func=";
  summary << activationFunctionToStr(_act_func);

  if (detailed && _sparsity < 1.0) {
    summary << ", sampling=(";
    buildSamplingSummary(summary);
    summary << ")";
  }

  summary << "\n";
}

void FullyConnectedLayer::buildSamplingSummary(std::ostream& summary) const {
  if (_sparsity < 1.0) {
    if (useRandomSampling()) {
      summary << "random";
    } else {
      summary << "hash_function=" << _hasher->getName() << ", ";
      _hash_table->summarize(summary);
    }
  }
}

}  // namespace thirdai::bolt