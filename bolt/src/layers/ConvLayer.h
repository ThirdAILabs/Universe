#pragma once

#include "FullyConnectedLayer.h"

namespace thirdai::bolt {
class ConvLayer : public FullyConnectedLayer {
 public:
  ConvLayer(const FullyConnectedLayerConfig& config, uint64_t prev_dim,
            uint32_t prev_num_filters, uint32_t prev_num_sparse_filters,
            uint32_t next_kernel_size);

  void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels);

  void backpropagate(BoltVector& input, BoltVector& output);

  void backpropagateInputLayer(BoltVector& input, BoltVector& output);

  void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  void buildHashTables();

  void reBuildHashFunction();

  BoltBatch createBatchState(const uint32_t batch_size, bool) const {
    bool is_dense = _sparse_dim == _dim;
    return BoltBatch(is_dense ? _dim : _sparse_dim, batch_size, is_dense);
  }

 private:
  template <bool DENSE, bool PREV_DENSE>
  void forwardImpl(const BoltVector& input, BoltVector& output);

  template <bool DENSE, bool PREV_DENSE>
  void selectActiveFilters(const BoltVector& input, BoltVector& output,
                           uint32_t in_patch, uint64_t out_patch,
                           uint32_t* active_filters);

  template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
  void backpropagateImpl(BoltVector& input, BoltVector& output);

  void buildPatchMaps(uint32_t next_kernel_size);

  uint32_t _patch_dim, _sparse_patch_dim, _num_patches, _num_filters,
      _num_sparse_filters, _prev_num_filters, _prev_num_sparse_filters,
      _kernel_size;
  std::vector<uint32_t> _in_to_out, _out_to_in;
};
}  // namespace thirdai::bolt