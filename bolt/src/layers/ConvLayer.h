#pragma once

#include "SequentialLayer.h"
#include <hashing/src/DWTA.h>
#include <hashtable/src/SampledHashTable.h>

namespace thirdai::bolt {
class ConvLayer : public SequentialLayer {
 public:
  ConvLayer(const FullyConnectedLayerConfig& config, uint64_t prev_dim,
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

  float* getWeights() final;

  float* getBiases() final;

  void setWeights(float* new_weights) final;

  void setBiases(float* new_biases) final;

 private:
  // can't be inlined .cc if part of an interface. see here:
  // https://stackoverflow.com/questions/27345284/is-it-possible-to-declare-constexpr-class-in-a-header-and-define-it-in-a-separat
  constexpr float actFuncDerivative(float x) {
    switch (_act_func) {
      case ActivationFunction::ReLU:
        return x > 0 ? 1.0 : 0.0;
      case ActivationFunction::Softmax:
        // return 1.0; // Commented out because Clang tidy doesn't like
        // consecutive identical branches
      case ActivationFunction::Linear:
        return 1.0;
        // default:
        //   return 0.0;
    }
    // This is impossible to reach, but the compiler gave a warning saying it
    // reached the end of a non void function without it.
    return 0.0;
  }

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

  void buildPatchMaps(std::tuple<uint32_t, uint32_t> next_kernel_size);

  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;
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

  uint32_t _patch_dim;         // the dim of a patch if the input was dense
  uint32_t _sparse_patch_dim;  // the actual dim of a patch
  uint32_t _num_patches;
  uint32_t _num_filters;         // number of convolutional filters
  uint32_t _num_sparse_filters;  // _num_filters * sparsity
  uint32_t _prev_num_filters;
  uint32_t _prev_num_sparse_filters;
  uint32_t _kernel_size;
  std::vector<uint32_t> _in_to_out, _out_to_in;  // patch mappings
};
}  // namespace thirdai::bolt