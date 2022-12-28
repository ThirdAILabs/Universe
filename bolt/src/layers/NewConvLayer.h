#pragma once

#include <cereal/types/vector.hpp>
#include "LayerConfig.h"
#include <bolt/src/layers/Optimizer.h>
#include <optional>
#include <random>
#include <stdexcept>

namespace thirdai::bolt {
class ConvLayer final {
 public:
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer(ConvLayer&&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;
  ConvLayer& operator=(ConvLayer&&) = delete;

  ConvLayer(const ConvLayerConfig& config, uint64_t prev_dim,
            uint32_t prev_num_filters, uint32_t prev_num_sparse_filters) {
    // TODO(david) initialize helper variables (with better names?), active
    // neuron trackers, etc

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::normal_distribution<float> dist(0.0, 0.01);

    std::generate(_weights.begin(), _weights.end(),
                  [&]() { return dist(eng); });
    std::generate(_biases.begin(), _biases.end(), [&]() { return dist(eng); });

    if (_sparsity < 1.0) {
      initSamplingDatastructures(config.getSamplingConfig(), rd);
    }

    initOptimizer();
  }

  void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels) {
    // TODO(david): eigen forward pass?
    if (output.isDense()) {
      if (input.isDense()) {
        forwardImpl</*DENSE=*/true, /*PREV_DENSE=*/true>(input, output);
      } else {
        forwardImpl</*DENSE=*/true, /*PREV_DENSE=*/false>(input, output);
      }
    } else {
      if (input.isDense()) {
        forwardImpl</*DENSE=*/false, /*PREV_DENSE=*/true>(input, output);
      } else {
        forwardImpl</*DENSE=*/false, /*PREV_DENSE=*/false>(input, output);
      }
    }
  }

  void backpropagate(BoltVector& input, BoltVector& output) {
    // TODO (david): evaluate eigen
    if (output.isDense()) {
      if (input.isDense()) {
        backpropagateImpl</*IS_INPUT=*/false, /*DENSE=*/true,
                          /*PREV_DENSE=*/true>(input, output);
      } else {
        backpropagateImpl</*IS_INPUT=*/false, /*DENSE=*/true,
                          /*PREV_DENSE=*/false>(input, output);
      }
    } else {
      if (input.isDense()) {
        backpropagateImpl</*IS_INPUT=*/false, /*DENSE=*/false,
                          /*PREV_DENSE=*/true>(input, output);
      } else {
        backpropagateImpl</*IS_INPUT=*/false, /*DENSE=*/false,
                          /*PREV_DENSE=*/false>(input, output);
      }
    }
  }

  void backpropagateInputLayer(BoltVector& input, BoltVector& output) {
    if (output.isDense()) {
      if (input.isDense()) {
        backpropagateImpl</*IS_INPUT=*/true, /*DENSE=*/true,
                          /*PREV_DENSE=*/true>(input, output);
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

  void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  BoltBatch createBatchState(const uint32_t batch_size,
                             bool use_sparsity) const {
    bool is_sparse = (_sparsity < 1.0) && use_sparsity;

    uint32_t curr_dim = is_sparse ? _sparse_dim : _dim;

    return BoltBatch(/* dim= */ curr_dim, /* batch_size= */ batch_size,
                     /* is_dense= */ !is_sparse);
  }

  // TODO(david): add freeze hash tables feature later and test with it.

  void buildHashTables() {
    if (_sparsity >= 1.0) {
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

  void reBuildHashFunction() {
    if (_sparsity >= 1.0) {
      return;
    }
    _hasher = _hasher->copyWithNewSeeds();
  }

  uint32_t getDim() const { return _dim; }

  uint32_t getInputDim() const { return _prev_dim; }

  uint32_t getSparseDim() const { return _sparse_dim; }

  float getSparsity() const { return _sparsity; }

  void setSparsity(float sparsity);

  ActivationFunction getActivationFunction() const { return _act_func; }

  void buildLayerSummary(std::stringstream& summary, bool detailed) const;

  void initOptimizer() {
    if (!_weight_optimizer || !_bias_optimizer) {
      _weight_optimizer = AdamOptimizer(_dim * _prev_dim);
      _bias_optimizer = AdamOptimizer(_dim);
    }
  }

  ~ConvLayer() = default;

 private:
  template <bool DENSE, bool PREV_DENSE>
  void forwardImpl(const BoltVector& input, BoltVector& output) {
    // TODO(david): add assert statements appropriate to conv here
  }

  template <bool PREV_DENSE>
  void selectActiveFilters(const BoltVector& input, BoltVector& output,
                           uint32_t in_patch, uint64_t out_patch,
                           const std::vector<uint32_t>& active_filters);

  template <bool DENSE, bool PREV_DENSE>
  float calculateFilterActivation(const BoltVector& input,
                                  const BoltVector& output, uint32_t in_patch,
                                  uint64_t out_idx,
                                  std::vector<uint32_t> prev_active_filters,
                                  uint32_t effective_patch_dim);

  template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
  void backpropagateImpl(BoltVector& input, BoltVector& output);

  void initSamplingDatastructures(const SamplingConfigPtr& sampling_config,
                                  std::random_device& rd) {
    if (sampling_config->isRandomSampling()) {
      throw std::invalid_argument(
          "Conv Layer does not currently support random sampling.");
    }

    // hashes input of size _patch_dim
    _hasher = sampling_config->getHashFunction(_patch_dim);

    _hash_table = sampling_config->getHashTable();

    buildHashTables();

    _rand_neurons = std::vector<uint32_t>(_num_filters);

    std::iota(_rand_neurons.begin(), _rand_neurons.end(), 0);
    std::shuffle(_rand_neurons.begin(), _rand_neurons.end(), rd);
  }

  void buildPatchMaps(std::pair<uint32_t, uint32_t> next_kernel_size);

  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;
  ActivationFunction _act_func;

  std::vector<float> _weights;
  std::vector<float> _biases;

  std::optional<AdamOptimizer> _weight_optimizer = std::nullopt;
  std::optional<AdamOptimizer> _bias_optimizer = std::nullopt;

  std::vector<bool> _is_active;

  std::unique_ptr<hashing::HashFunction> _hasher;
  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> _hash_table;
  std::vector<uint32_t> _rand_neurons;

  uint32_t _num_filters;         // number of convolutional filters
  uint32_t _num_sparse_filters;  // _num_filters * sparsity
  uint32_t _patch_dim;           // the dim of a patch if the input was dense
  uint32_t _sparse_patch_dim;    // the actual dim of a patch
  uint32_t _num_patches;
  uint32_t _prev_num_filters;
  uint32_t _prev_num_sparse_filters;
  uint32_t _kernel_size;
  std::vector<uint32_t> _in_to_out, _out_to_in;  // patch mappings

  /**
   * Training data-structures (like the optimizer and the active neurons
   * trackers) are not loaded in by default. If we want to continue training
   * after a load, the expectation is that the higher level Graph/Network API
   * will handle this initialization with the initOptimizer() method.
   *
   * Doing this means our load API is as simple as possible for both
   * training and inference purposes. It doesn't make sense to load these
   * data-structures by default then remove them with another function since
   * users may be memory constrained during deployment.
   *
   * We don't know yet if its worth it to save the optimizer for
   * retraining/finetuning purposes. If in the future we figure out this has
   * some benefit we can adjust this method accordingly.
   */
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
            _biases, _is_active, _hasher, _hash_table, _rand_neurons,
            _patch_dim, _sparse_patch_dim, _num_patches, _num_filters,
            _num_sparse_filters, _prev_num_filters, _prev_num_sparse_filters,
            _kernel_size, _in_to_out, _out_to_in);
  }

 protected:
  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  ConvLayer() {}
};
}  // namespace thirdai::bolt
