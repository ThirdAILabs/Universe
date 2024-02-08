#include "FullyConnectedLayer.h"
#include <wrappers/src/EigenDenseWrapper.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/neuron_index/LshIndex.h>
#include <hashing/src/DWTA.h>
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/util/Constants.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <archive/src/ParameterReference.h>
#include <utils/Random.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::bolt {

FullyConnectedLayer::FullyConnectedLayer(
    const FullyConnectedLayerConfig& config, uint64_t prev_dim,
    bool disable_sparse_parameter_updates, bool use_bias)
    : _dim(config.getDim()),
      _prev_dim(prev_dim),
      _sparse_dim(config.getSparsity() * config.getDim()),
      _sparsity(config.getSparsity()),
      _act_func(config.getActFunc()),
      _weights(config.getDim() * prev_dim),
      _biases(config.getDim()),
      _disable_sparse_parameter_updates(disable_sparse_parameter_updates),
      _should_save_optimizer(false),
      _use_bias(use_bias),
      _prev_is_active(prev_dim, false),
      _is_active(config.getDim(), false) {
  std::mt19937 rng(global_random::nextSeed());
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(_weights.begin(), _weights.end(), [&]() { return dist(rng); });

  if (_use_bias) {
    std::generate(_biases.begin(), _biases.end(), [&]() { return dist(rng); });
  } else {
    _biases.assign(_biases.size(), 0.0);
  }
  if (_sparsity < 1.0) {
    _neuron_index = config.getSamplingConfig()->getNeuronIndex(_dim, _prev_dim);

    buildHashTables();
  }

  initActiveNeuronsTrackers();
}

FullyConnectedLayer::FullyConnectedLayer(const ar::Archive& archive)
    : _dim(archive.u64("dim")),
      _prev_dim(archive.u64("input_dim")),
      _sparse_dim(archive.u64("dim") * archive.getAs<ar::F32>("sparsity")),
      _sparsity(archive.getAs<ar::F32>("sparsity")),
      _act_func(bolt::getActivationFunction(archive.str("activation"))),
      _weights(archive.get("weights")->param().moveLoadedParameter()),
      _biases(archive.get("biases")->param().moveLoadedParameter()),
      _neuron_index(nullptr),
      _index_frozen(archive.boolean("index_frozen")),
      _disable_sparse_parameter_updates(
          archive.boolean("disable_sparse_parameter_updates")),
      _use_bias(archive.boolean("use_bias")) {
  if (archive.contains("neuron_index")) {
    _neuron_index = NeuronIndex::fromArchive(*archive.get("neuron_index"));
  }
  if (!_neuron_index && _sparsity < 1.0) {
    _neuron_index = DWTASamplingConfig::autotune(
                        _dim, _sparsity, /*experimental_autotune=*/false)
                        ->getNeuronIndex(_dim, _prev_dim);
  }

  if (archive.contains("weight_optimizer")) {
    _weight_optimizer = optimizerFromArchive(*archive.get("weight_optimizer"));
  }
  if (archive.contains("bias_optimizer")) {
    _bias_optimizer = optimizerFromArchive(*archive.get("bias_optimizer"));
  }

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

  if constexpr (!DENSE) {
    assert(_neuron_index);
    _neuron_index->query(input, output, labels);
  } else {
    (void)labels;
  }

  float max_act = 0;
  uint32_t len_out = nonzerosInOutput<DENSE>();

  // TODO(david) this is not needed for inference, we can optionally remove
  // this with some refactoring if we want slightly faster inference. This
  // function should be done in the backpropagate method, then we only mark
  // neurons for gradient updates when we actually compute gradients for them.
  markActiveNeuronsForUpdate<DENSE, PREV_DENSE>(input, output, len_out);

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
      eigen_output = (1 + (-eigen_output.array()).exp()).inverse();
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

void FullyConnectedLayer::updateParameters(float lr, uint32_t iter, float B1,
                                           float B2, float eps) {
  float B1_bias_corrected = static_cast<float>(1 - pow(B1, iter));
  float B2_bias_corrected = static_cast<float>(1 - pow(B2, iter));

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
  if (!_use_bias) {
    return;
  }
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

void FullyConnectedLayer::buildHashTables() {
  if (_sparsity >= 1.0 || _index_frozen) {
    return;
  }

  assert(_neuron_index);
  _neuron_index->buildIndex(_weights, _dim, /* use_new_seed= */ false);
}

void FullyConnectedLayer::reBuildHashFunction() {
  if (_sparsity >= 1.0 || _index_frozen) {
    return;
  }

  assert(_neuron_index);
  _neuron_index->buildIndex(_weights, _dim, /* use_new_seed= */ true);
}

void FullyConnectedLayer::setNeuronIndex(NeuronIndexPtr index) {
  _neuron_index = std::move(index);
  if (_neuron_index) {
    _neuron_index->buildIndex(_weights, _dim, /* use_new_seed= */ false);
  }
}

void FullyConnectedLayer::freezeHashTables(bool insert_labels_if_not_found) {
  _index_frozen = true;

  if (insert_labels_if_not_found && _neuron_index) {
    _neuron_index->insertLabelsIfNotFound();
  }
}

float* FullyConnectedLayer::getWeights() const {
  float* weights_copy = new float[_dim * _prev_dim];
  std::copy(_weights.begin(), _weights.end(), weights_copy);

  return weights_copy;
}

float* FullyConnectedLayer::getBiases() const {
  float* biases_copy = new float[_dim];
  std::copy(_biases.begin(), _biases.end(), biases_copy);

  return biases_copy;
}

void FullyConnectedLayer::setWeights(const float* new_weights) {
  std::copy(new_weights, new_weights + _dim * _prev_dim, _weights.begin());

  buildHashTables();
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

void FullyConnectedLayer::setSparsity(float sparsity, bool rebuild_hash_tables,
                                      bool experimental_autotune) {
  // TODO(Nick): Right now we always switch to DWTA after setting sparsity
  // instead of autotuning for whatever the existing hash function was. We
  // should instead autotune the original hash function.

  thirdai::bolt::checkSparsity(sparsity);

  if (_sparsity == 1 && sparsity < 1) {
    _sparsity = sparsity;
    _sparse_dim = _sparsity * _dim;

    std::mt19937 rng(global_random::nextSeed());
    _neuron_index =
        DWTASamplingConfig::autotune(_dim, sparsity, experimental_autotune)
            ->getNeuronIndex(_dim, _prev_dim);
    return;
  }

  if (_sparsity < 1 && sparsity == 1) {
    _sparsity = 1;
    _sparse_dim = _dim;
    _neuron_index = {};
    return;
  }

  if (_sparsity < 1 && sparsity < 1) {
    _sparsity = sparsity;
    _sparse_dim = _sparsity * _dim;

    assert(_neuron_index);
    if (rebuild_hash_tables) {
      _neuron_index->autotuneForNewSparsity(_dim, _prev_dim, sparsity,
                                            experimental_autotune);
    }
    return;
  }
}

std::pair<hashing::HashFunctionPtr, hashtable::SampledHashTablePtr>
FullyConnectedLayer::getHashTable() {
  if (auto index = LshIndex::cast(_neuron_index)) {
    return {index->hashFn(), index->hashTable()};
  }

  return {nullptr, nullptr};
}

void FullyConnectedLayer::setHashTable(
    hashing::HashFunctionPtr hash_fn,
    hashtable::SampledHashTablePtr hash_table) {
  if (hash_fn->numTables() != hash_table->numTables()) {
    throw std::invalid_argument(
        "Hash function returning " + std::to_string(hash_fn->numTables()) +
        "hashes cannot be used used with a hash table with " +
        std::to_string(hash_table->numTables()) + " tables.");
  }

  if (hash_fn->range() != hash_table->tableRange()) {
    throw std::invalid_argument("Hash function with range " +
                                std::to_string(hash_fn->range()) +
                                " cannot be used with hash table with range " +
                                std::to_string(hash_table->tableRange()) + ".");
  }

  uint32_t max_element = hash_table->maxElement();
  if (max_element >= _dim) {
    throw std::invalid_argument(
        "Hash table containing neuron index " + std::to_string(max_element) +
        " cannot be used in fully connected layer with dimension " +
        std::to_string(_dim) + ".");
  }

  _neuron_index =
      LshIndex::make(_dim, std::move(hash_fn), std::move(hash_table));
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
  if (_neuron_index) {
    _neuron_index->summarize(summary);
  }
}

}  // namespace thirdai::bolt