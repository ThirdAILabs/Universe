#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include <exceptions/src/Exceptions.h>
#include <numeric>
#include <random>

namespace thirdai::bolt {

ConvLayer::ConvLayer(const ConvLayerConfig& config, uint64_t prev_dim,
                     uint32_t prev_num_filters,
                     uint32_t prev_num_sparse_filters,
                     std::pair<uint32_t, uint32_t> next_kernel_size)
    : _dim(config.num_filters * config.num_patches),
      _prev_dim(prev_dim),
      _sparse_dim(config.sparsity * config.num_filters * config.num_patches),
      _sparsity(config.sparsity),
      _act_func(config.act_func),
      _sampling_config(config.sampling_config),
      _force_sparse_for_inference(false),
      _num_filters(config.num_filters),
      _num_sparse_filters(config.num_filters * config.sparsity),
      _num_patches(config.num_patches),
      _prev_num_filters(prev_num_filters),
      _prev_num_sparse_filters(prev_num_sparse_filters),
      _kernel_size(config.kernel_size.first * config.kernel_size.second) {
  if (_act_func != ActivationFunction::ReLU) {
    throw std::invalid_argument(
        "Conv layers currently support only ReLU Activation.");
  }

  if (config.kernel_size.first != config.kernel_size.second) {
    throw std::invalid_argument(
        "Conv layers currently support only square kernels.");
  }

  _patch_dim = _kernel_size * _prev_num_filters;
  _sparse_patch_dim = _kernel_size * _prev_num_sparse_filters;

  _weights = std::vector<float>(_num_filters * _patch_dim);
  _w_gradient = std::vector<float>(_num_filters * _patch_dim, 0);
  _w_momentum = std::vector<float>(_num_filters * _patch_dim, 0);
  _w_velocity = std::vector<float>(_num_filters * _patch_dim, 0);

  _biases = std::vector<float>(_num_filters);
  _b_gradient = std::vector<float>(_num_filters, 0);
  _b_momentum = std::vector<float>(_num_filters, 0);
  _b_velocity = std::vector<float>(_num_filters, 0);

  _is_active = std::vector<bool>(_num_filters * _num_patches, false);

  buildPatchMaps(next_kernel_size);

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(_weights.begin(), _weights.end(), [&]() { return dist(eng); });
  std::generate(_biases.begin(), _biases.end(), [&]() { return dist(eng); });

  if (_sparsity < 1.0) {
    // hashes input of size _patch_dim
    _hasher = std::make_unique<hashing::DWTAHashFunction>(
        _patch_dim, _sampling_config.hashes_per_table,
        _sampling_config.num_tables, _sampling_config.range_pow);

    _hash_table = std::make_unique<hashtable::SampledHashTable<uint32_t>>(
        _sampling_config.num_tables, _sampling_config.reservoir_size,
        1 << _sampling_config.range_pow);

    buildHashTables();

    _rand_neurons = std::vector<uint32_t>(_num_filters);

    std::iota(_rand_neurons.begin(), _rand_neurons.end(), 0);
    std::shuffle(_rand_neurons.begin(), _rand_neurons.end(), rd);
  }
}

void ConvLayer::forward(const BoltVector& input, BoltVector& output,
                        const BoltVector* labels) {
  (void)labels;
  if (output.isDense()) {
    if (input.isDense()) {
      forwardImpl<true, true>(input, output);
    } else {
      forwardImpl<true, false>(input, output);
    }
  } else {
    if (input.isDense()) {
      forwardImpl<false, true>(input, output);
    } else {
      forwardImpl<false, false>(input, output);
    }
  }
}

template <bool DENSE, bool PREV_DENSE>
void ConvLayer::forwardImpl(const BoltVector& input, BoltVector& output) {
  uint32_t num_active_filters;
  if (DENSE) {
    num_active_filters = _num_filters;
    std::fill_n(output.gradients, _dim, 0);
  } else {
    num_active_filters = _num_sparse_filters;
    std::fill_n(output.gradients, _sparse_dim, 0);
  }

  // elements to loop through to calculate a dot product filter act on a patch
  uint32_t effective_patch_dim = PREV_DENSE ? _patch_dim : _sparse_patch_dim;

  // input.active_neurons[i] is an index into input.activations with an offset
  // we mod here to remove that offset
  std::vector<uint32_t> prev_active_filters(input.len);  // unused if DENSE
  if (!PREV_DENSE) {
    // TODO(david) calculate once instead of in both forward and backward?
    for (uint32_t i = 0; i < input.len; i++) {
      prev_active_filters[i] = input.active_neurons[i] % _patch_dim;
    }
  }

  // for each patch, we look at a section of the input (an input patch) and
  // populate a section of the output (an output patch) with that input's
  // filter activations. in_patch and out_patch tell us what patch we are
  // looking at in the input/output respectively
  for (uint32_t in_patch = 0; in_patch < _num_patches; in_patch++) {
    uint32_t out_patch = _in_to_out[in_patch];

    if (!DENSE) {
      selectActiveFilters<PREV_DENSE>(input, output, in_patch, out_patch,
                                      prev_active_filters);
    }

    // for each filter selected
    for (uint32_t filter = 0; filter < num_active_filters; filter++) {
      // out_idx: the actual INDEX in the output that corresponds to where a
      // patch's filter activations lie
      uint64_t out_idx = out_patch * num_active_filters + filter;
      float act = calculateFilterActivation<DENSE, PREV_DENSE>(
          input, output, in_patch, out_idx, prev_active_filters,
          effective_patch_dim);
      assert(!std::isnan(act));
      output.activations[out_idx] = std::max(0.0F, act);
    }
  }
}

template <bool DENSE, bool PREV_DENSE>
float ConvLayer::calculateFilterActivation(
    const BoltVector& input, const BoltVector& output, uint32_t in_patch,
    uint64_t out_idx, std::vector<uint32_t> prev_active_filters,
    uint32_t effective_patch_dim) {
  uint64_t act_neuron = DENSE ? out_idx : output.active_neurons[out_idx];
  assert(act_neuron < _dim);

  uint32_t act_filter = act_neuron % _num_filters;  // remove offset again

  _is_active[act_neuron] = true;  // used in updateParameters

  // calculate filter activation via dot product
  float act = _biases[act_filter];
  for (uint32_t i = 0; i < effective_patch_dim; i++) {
    uint64_t in_idx = in_patch * effective_patch_dim + i;
    assert(in_idx < input.len);
    uint64_t prev_act_neuron = PREV_DENSE ? i : prev_active_filters[in_idx];

    act += _weights[act_filter * _patch_dim + prev_act_neuron] *
           input.activations[in_idx];
  }
  return act;
}

void ConvLayer::backpropagate(BoltVector& input, BoltVector& output) {
  if (output.isDense()) {
    if (input.isDense()) {
      backpropagateImpl<false, true, true>(input, output);
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

void ConvLayer::backpropagateInputLayer(BoltVector& input, BoltVector& output) {
  if (output.isDense()) {
    if (input.isDense()) {
      backpropagateImpl<true, true, true>(input, output);
    } else {
      backpropagateImpl<true, true, false>(input, output);
    }
  } else {
    if (input.isDense()) {
      backpropagateImpl<true, false, true>(input, output);
    } else {
      backpropagateImpl<true, false, false>(input, output);
    }
  }
}

template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
void ConvLayer::backpropagateImpl(BoltVector& input, BoltVector& output) {
  uint32_t len_out = DENSE ? _dim : _sparse_dim;
  uint32_t num_active_filters = DENSE ? _num_filters : _num_sparse_filters;
  uint32_t effective_patch_dim = PREV_DENSE ? _patch_dim : _sparse_patch_dim;

  std::vector<uint32_t> prev_active_filters(input.len);  // unused if DENSE
  if (!PREV_DENSE) {
    for (uint32_t i = 0; i < input.len; i++) {
      prev_active_filters[i] = input.active_neurons[i] % _patch_dim;
    }
  }

  // loop through every output neuron
  for (uint64_t n = 0; n < len_out; n++) {
    assert(!std::isnan(output.gradients[n]));
    output.gradients[n] *= actFuncDerivative(output.activations[n], _act_func);
    assert(!std::isnan(output.gradients[n]));

    uint32_t act_neuron = DENSE ? n : output.active_neurons[n];
    uint32_t act_filter = act_neuron % _num_filters;

    uint32_t out_patch = n / num_active_filters;
    uint32_t in_patch = _out_to_in[out_patch];

    // loop through each input neuron and update the gradients
    for (uint64_t i = 0; i < effective_patch_dim; i++) {
      uint64_t in_idx = in_patch * effective_patch_dim + i;
      uint32_t prev_act_neuron = PREV_DENSE ? i : prev_active_filters[in_idx];
      assert(prev_act_neuron < _prev_dim);
      _w_gradient[act_filter * _patch_dim + prev_act_neuron] +=
          output.gradients[n] * input.activations[in_idx];
      if (!FIRST_LAYER) {
        input.gradients[in_idx] +=
            output.gradients[n] *
            _weights[act_filter * _patch_dim + prev_act_neuron];
      }
    }
    _b_gradient[act_filter] += output.gradients[n];
  }
}

template <bool PREV_DENSE>
void ConvLayer::selectActiveFilters(
    const BoltVector& input, BoltVector& output, uint32_t in_patch,
    uint64_t out_patch, const std::vector<uint32_t>& prev_active_filters) {
  // hash a section of the input (the input patch) and populate a section of the
  // output (the output patch) with that input's active filters (with an offset)
  std::unordered_set<uint32_t> active_set;
  std::vector<uint32_t> hashes(_hasher->numTables());
  if (PREV_DENSE) {
    _hasher->hashSingleDense(&input.activations[in_patch * _patch_dim],
                             _patch_dim, hashes.data());
  } else {
    _hasher->hashSingleSparse(
        prev_active_filters.data() + in_patch * _sparse_patch_dim,
        &input.activations[in_patch * _sparse_patch_dim], _sparse_patch_dim,
        hashes.data());
  }
  _hash_table->queryBySet(hashes.data(), active_set);

  if (active_set.size() < _num_sparse_filters) {
    uint32_t rand_offset = rand() % _num_filters;
    while (active_set.size() < _num_sparse_filters) {
      active_set.insert(_rand_neurons[rand_offset++]);
      rand_offset = rand_offset % _num_filters;
    }
  }

  uint32_t cnt = 0;
  for (uint32_t x : active_set) {
    if (cnt == _num_sparse_filters) {
      break;
    }
    assert(x < _num_filters);
    output.active_neurons[out_patch * _num_sparse_filters + cnt] =
        out_patch * _num_filters + x;
    cnt++;
  }
}

void ConvLayer::updateParameters(float lr, uint32_t iter, float B1, float B2,
                                 float eps) {
  float B1_bias_corrected = static_cast<float>(1 - pow(B1, iter));
  float B2_bias_corrected = static_cast<float>(1 - pow(B2, iter));

#pragma omp parallel for default(none) \
    shared(lr, B1, B1_bias_corrected, B2, B2_bias_corrected, eps)
  for (uint64_t n = 0; n < _num_filters; n++) {
    if (!_is_active[n]) {
      continue;
    }

    for (uint64_t i = 0; i < _patch_dim; i++) {
      auto indx = n * _patch_dim + i;
      float grad = _w_gradient[indx];
      assert(!std::isnan(grad));

      _w_momentum[indx] = B1 * _w_momentum[indx] + (1 - B1) * grad;
      _w_velocity[indx] = B2 * _w_velocity[indx] + (1 - B2) * grad * grad;
      assert(!std::isnan(_w_momentum[indx]));
      assert(!std::isnan(_w_velocity[indx]));

      _weights[indx] +=
          lr * (_w_momentum[indx] / B1_bias_corrected) /
          (std::sqrt(_w_velocity[indx] / B2_bias_corrected) + eps);
      assert(!std::isnan(_weights[indx]));

      _w_gradient[indx] = 0;
    }

    float grad = _b_gradient[n];
    assert(!std::isnan(grad));

    _b_momentum[n] = B1 * _b_momentum[n] + (1 - B1) * grad;
    _b_velocity[n] = B2 * _b_velocity[n] + (1 - B2) * grad * grad;

    assert(!std::isnan(_b_momentum[n]));
    assert(!std::isnan(_b_velocity[n]));

    _biases[n] += lr * (_b_momentum[n] / B1_bias_corrected) /
                  (std::sqrt(_b_velocity[n] / B2_bias_corrected) + eps);
    assert(!std::isnan(_biases[n]));

    _b_gradient[n] = 0;
    _is_active[n] = false;
  }
}

void ConvLayer::reBuildHashFunction() {
  if (_sparsity >= 1.0 || _force_sparse_for_inference) {
    return;
  }
  _hasher = std::make_unique<hashing::DWTAHashFunction>(
      _patch_dim, _sampling_config.hashes_per_table,
      _sampling_config.num_tables, _sampling_config.range_pow);
}

void ConvLayer::buildHashTables() {
  if (_sparsity >= 1.0 || _force_sparse_for_inference) {
    return;
  }
  uint64_t num_tables = _hash_table->numTables();
  std::vector<uint32_t> hashes(num_tables * _num_filters);
#pragma omp parallel for default(none) shared(num_tables, hashes)
  for (uint64_t n = 0; n < _num_filters; n++) {
    _hasher->hashSingleDense(_weights.data() + n * _patch_dim, _patch_dim,
                             hashes.data() + n * num_tables);
  }

  _hash_table->clearTables();
  _hash_table->insertSequential(_num_filters, 0, hashes.data());
}

void ConvLayer::shuffleRandNeurons() {
  if (_sparsity < 1.0 && !_force_sparse_for_inference) {
    std::shuffle(_rand_neurons.begin(), _rand_neurons.end(),
                 std::random_device{});
  }
}

float* ConvLayer::getWeights() const {
  float* weights_copy = new float[_dim * _prev_dim];
  std::copy(_weights.begin(), _weights.end(), weights_copy);

  return weights_copy;
}

float* ConvLayer::getBiases() const {
  float* biases_copy = new float[_dim];
  std::copy(_biases.begin(), _biases.end(), biases_copy);

  return biases_copy;
}

void ConvLayer::setTrainable(bool trainable) {
  (void)trainable;
  throw thirdai::exceptions::NotImplemented(
      "setTrainable not implemented for ConvLayer");
}

bool ConvLayer::getTrainable() const {
  throw thirdai::exceptions::NotImplemented(
      "getTrainable not implemented for ConvLayer");
}

void ConvLayer::setWeights(const float* new_weights) {
  std::copy(new_weights, new_weights + _dim * _prev_dim, _weights.begin());
}

void ConvLayer::setBiases(const float* new_biases) {
  std::copy(new_biases, new_biases + _dim, _biases.begin());
}

void ConvLayer::setWeightGradients(const float* update_weight_gradient) {
  std::copy(update_weight_gradient, update_weight_gradient + _dim,
            _w_gradient.begin());
}

void ConvLayer::setBiasesGradients(const float* update_bias_gradient) {
  std::copy(update_bias_gradient, update_bias_gradient + _dim,
            _b_gradient.begin());
}

float* ConvLayer::getBiasesGradient() {
  float* biases_gradients_copy = new float[_dim];
  std::copy(_b_gradient.begin(), _b_gradient.end(), biases_gradients_copy);

  return biases_gradients_copy;
}

float* ConvLayer::getWeightsGradient() {
  float* weights_gradients_copy = new float[_dim];
  std::copy(_w_gradient.begin(), _w_gradient.end(), weights_gradients_copy);

  return weights_gradients_copy;
}

// this function is only called from constructor
void ConvLayer::buildPatchMaps(std::pair<uint32_t, uint32_t> next_kernel_size) {
  /** TODO(David): btw this will be factored out soon into an N-tower model and
  a patch remapping

  Suppose we have an image that can be broken into 16 patches (numbers 0-15)
  with kernel size K x K as follows:
    0  1  2  3
    4  5  6  7
    8  9  10 11
    12 13 14 15

  Suppose then that each patch is flattened into a (K x K x channels)
  dimensional vector and concatenated next to each other in patch order (patch
  0, patch 1, ...). This is the input to the ConvLayer.

  To prepare for a possible future ConvLayer, we need to
  distribute the patches in memory in a way that is amenable for another
  convolution. Therefore, we accept a next_kernel_size and calculate an in_patch
  to out_patch mapping. The patch output needs to be placed somewhere in memory
  that is efficient for the next kernel size. Since we are dealing with
  flattened images, we need to remap the patches such that all patches within a
  future kernel are next to each other.

  Take this for example, where we are preparing for a future 2x2 kernel_size. We
  have patches 0, 1, 4, 5 that all will be together in a future 2x2 kernel and
  should be placed next to each other for the next ConvLayer. However, since
  their indices are not adjacent, they won't be placed next to each other. Thus
  we must remap them. Under this remapping, the 2x2 kernels in the next layer
  will each contain the 4 patch outputs necessary for the kernels to work.

    0  1  2  3           0  1  4  5
    4  5  6  7    -->    2  3  6  7
    8  9  10 11          8  9  12 13
    12 13 14 15          10 11 14 15
  **/
  if (next_kernel_size.first != next_kernel_size.second) {
    throw std::invalid_argument(
        "Conv layers currently support only square kernels.");
  }

  _in_to_out = std::vector<uint32_t>(_num_patches);
  _out_to_in = std::vector<uint32_t>(_num_patches);

  uint32_t next_filter_length = next_kernel_size.first;
  uint32_t num_patches_for_side =
      std::sqrt(_num_patches);  // assumes square images

  // this is a vector of the top left patch values for the next kernel size. in
  // the example above, this vector would be <0, 2, 8, 10>
  std::vector<uint32_t> top_left_patch_vals;
  for (uint32_t i = 0;
       i <= _num_patches - next_filter_length -
                ((next_filter_length - 1) * num_patches_for_side);
       i += next_filter_length) {
    top_left_patch_vals.push_back(i);
    if (((i + next_filter_length) % num_patches_for_side) == 0) {
      i += (next_filter_length - 1) * num_patches_for_side;
    }
  }

  uint32_t patch = 0;
  // for each top left patch val, map the patches in order to the values within
  // the filter for that patch val
  for (uint32_t start : top_left_patch_vals) {
    uint32_t base_val = start;
    for (uint32_t y = 0; y < next_filter_length; y++) {
      for (uint32_t x = 0; x < next_filter_length; x++) {
        uint32_t new_patch = base_val + x;
        _in_to_out[patch] = new_patch;
        _out_to_in[new_patch] = patch;
        patch++;
      }
      base_val += num_patches_for_side;
    }
  }
}
}  // namespace thirdai::bolt