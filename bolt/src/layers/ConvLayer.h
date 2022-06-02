#pragma once

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include "LayerConfig.h"
#include "LayerUtils.h"
#include "SequentialLayer.h"
#include <hashing/src/DWTA.h>
#include <hashtable/src/SampledHashTable.h>

namespace thirdai::bolt {
class ConvLayer final : public SequentialLayer {
 public:
  ConvLayer(const ConvLayerConfig& config, uint64_t prev_dim,
            uint32_t prev_num_filters, uint32_t prev_num_sparse_filters,
            std::pair<uint32_t, uint32_t> next_kernel_size);

  void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels) final;

  void backpropagate(BoltVector& input, BoltVector& output) final;

  void backpropagateInputLayer(BoltVector& input, BoltVector& output) final;

  void updateParameters(float lr, uint32_t iter, float B1, float B2,
                        float eps) override;

  BoltBatch createBatchState(const uint32_t batch_size,
                             bool force_dense) const final {
    bool is_dense = (_sparse_dim == _dim) || force_dense;

    return BoltBatch(is_dense ? _dim : _sparse_dim, batch_size, is_dense);
  }

  void forceSparseForInference() final {
    if (_sparsity < 1.0) {
      _force_sparse_for_inference = true;
    }
  }

  bool isForceSparsity() const final { return _force_sparse_for_inference; }

  void buildHashTables() final;

  void reBuildHashFunction() final;

  void shuffleRandNeurons() final;

  uint32_t getDim() const final { return _dim; }

  uint32_t getInputDim() const final { return _prev_dim; }

  uint32_t getInferenceOutputDim() const final {
    if (_force_sparse_for_inference) {
      return _sparse_dim;
    }
    return _dim;
  }

  float* getWeights() final;

  float* getBiases() final;

  void setWeights(const float* new_weights) final;

  void setBiases(const float* new_biases) final;

  void setShallowSave(bool set) final {
    (void)set;
    // throw std::logic_error("Warning: setShallowSave not implemented for
    // DLRM;");
  }

  void setShallow(bool set) final {
    (void)set;
    // throw std::logic_error("Warning: setShallow not implemented for DLRM;");
  }

  bool isShallow() final {
    // throw std::logic_error("Warning: isShallow not implemented for DLRM;");
    return false;
  }

  void initOptimizer() final {
    (void)true;
    // throw std::logic_error("Warning: initOptimizer not implemented for
    // DLRM;");
  }

  void remOptimizer() final {
    (void)true;
    // throw std::logic_error("Warning: remOptimizer not implemented for
    // DLRM;");
  }

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

  void buildPatchMaps(std::pair<uint32_t, uint32_t> next_kernel_size);

  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;
  bool _shallow_save;
  ActivationFunction _act_func;

  std::vector<float> _weights;
  std::vector<float> _w_gradient;
  std::vector<float> _w_momentum;
  std::vector<float> _w_velocity;

  std::vector<float> _biases;
  std::vector<float> _b_gradient;
  std::vector<float> _b_momentum;
  std::vector<float> _b_velocity;

  std::vector<bool> _is_active;

  SamplingConfig _sampling_config;
  std::unique_ptr<hashing::DWTAHashFunction> _hasher;
  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> _hash_table;
  std::vector<uint32_t> _rand_neurons;

  bool _force_sparse_for_inference;

  uint32_t _num_filters;         // number of convolutional filters
  uint32_t _num_sparse_filters;  // _num_filters * sparsity
  uint32_t _patch_dim;           // the dim of a patch if the input was dense
  uint32_t _sparse_patch_dim;    // the actual dim of a patch
  uint32_t _num_patches;
  uint32_t _prev_num_filters;
  uint32_t _prev_num_sparse_filters;
  uint32_t _kernel_size;
  std::vector<uint32_t> _in_to_out, _out_to_in;  // patch mappings

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
            _w_gradient, _w_momentum, _w_velocity, _biases, _b_gradient,
            _b_momentum, _b_velocity, _is_active, _sampling_config, _hasher,
            _hash_table, _rand_neurons, _force_sparse_for_inference, _patch_dim,
            _sparse_patch_dim, _num_patches, _num_filters, _num_sparse_filters,
            _prev_num_filters, _prev_num_sparse_filters, _kernel_size,
            _in_to_out, _out_to_in);
  }

 protected:
  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  ConvLayer() {}
};
}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::ConvLayer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(thirdai::bolt::SequentialLayer,
                                     thirdai::bolt::ConvLayer)
