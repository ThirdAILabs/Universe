#pragma once

#include <cereal/types/vector.hpp>
#include "LayerConfig.h"
#include "LayerUtils.h"
#include <bolt/src/layers/Optimizer.h>
#include <hashing/src/DWTA.h>
#include <hashtable/src/SampledHashTable.h>
#include <exceptions/src/Exceptions.h>
#include <optional>
#include <random>
#include <stdexcept>

namespace thirdai::bolt {

/**
 * This layer implements a typical convolution but with one notable exception:
 * adding sparsity in the channels. More specifically, each patch in the input
 * is hashed to a set of convolutional filters where we select the "sparsity"
 * percent of these filters that are most similar to the input patch as defined
 * by LSH.
 *
 * Currently, inputs are assumed to be BoltVectors where patches are first
 * flattened then appended to the BoltVector going first horizontally (by patch)
 * then vertically. As an exmaple, if an image has 4 patches numbered 1, 2, 3, 4
 * in this configuration:
 *                        | 1 2 |
 *                        | 3 4 |
 * The patches would be layed out in numerical order. For further clarification,
 * suppose that same image had a dimension of 8 x 8 x 3. Each patch would be of
 * size 4 x 4 x 3, would be flattened beforehand, and added (in numerical order)
 * to the input BoltVector. In the event of a future conv layer, we remap the
 * output patches to suitable locations (more info can be found in the
 * buildPatchMaps function). This input format is likely ineffecient and will be
 * refactored later.
 *
 * Finally, the output of this layer can be either sparse or dense. In the event
 * of a sparse output, the order or patch information is the same but the output
 * channels now are represented in a sparse manner by index, value pairs. These
 * index value pairs are offset by their patch * num_filters so they can be used
 * by future bolt layers normally.
 */
class ConvLayer final {
 public:
  ConvLayer(const ConvLayerConfig& config, uint32_t prev_height,
            uint32_t prev_width, uint32_t prev_num_channels,
            float prev_sparsity);

  void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels);

  void backpropagate(BoltVector& input, BoltVector& output);

  void backpropagateInputLayer(BoltVector& input, BoltVector& output);

  void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  BoltBatch createBatchState(const uint32_t batch_size,
                             bool use_sparsity) const {
    bool is_sparse = (_sparsity < 1.0) && use_sparsity;

    uint32_t curr_dim = is_sparse ? _sparse_dim : _dim;

    return BoltBatch(/* dim= */ curr_dim, /* batch_size= */ batch_size,
                     /* is_dense= */ !is_sparse);
  }

  void buildHashTables();

  void reBuildHashFunction();

  uint32_t getDim() const { return _dim; }

  uint32_t getInputDim() const { return _prev_dim; }

  uint32_t getSparseDim() const { return _sparse_dim; }

  std::tuple<uint32_t, uint32_t, uint32_t> getOutputDim3D() const {
    return std::make_tuple(_height, _width, _num_filters);
  }

  float* getWeights() const;

  float* getBiases() const;

  void setWeights(const float* new_weights);

  void setBiases(const float* new_biases);

  void setWeightGradients(const float* update_weight_gradient);

  void setBiasesGradients(const float* update_bias_gradient);

  float* getBiasesGradient();

  float* getWeightsGradient();

  float getSparsity() const { return _sparsity; }

  void initOptimizer();

  void buildLayerSummary(std::stringstream& summary) const;

  ~ConvLayer() = default;

 private:
  template <bool DENSE, bool PREV_DENSE>
  void forwardImpl(const BoltVector& input, BoltVector& output);

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
                                  std::random_device& rd);

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
  uint32_t _height, _width;
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
            _num_filters, _num_sparse_filters, _patch_dim, _sparse_patch_dim,
            _num_patches, _prev_num_filters, _prev_num_sparse_filters,
            _kernel_size, _height, _width, _in_to_out, _out_to_in);
  }

  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  ConvLayer() {}
};

}  // namespace thirdai::bolt
