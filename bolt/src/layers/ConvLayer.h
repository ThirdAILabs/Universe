#pragma once

#include "FullyConnectedLayer.h"
#include <tuple>

namespace thirdai::bolt {
class ConvLayer : public FullyConnectedLayer {
 public:
  ConvLayer(const FullyConnectedLayerConfig& config, uint64_t prev_dim,
            uint32_t prev_num_filters, uint32_t prev_num_sparse_filters,
            std::pair<uint32_t, uint32_t> next_kernel_size);

  void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels) override;

  void backpropagate(BoltVector& input, BoltVector& output) override;

  void backpropagateInputLayer(BoltVector& input, BoltVector& output) override;

  void updateParameters(float lr, uint32_t iter, float B1, float B2,
                        float eps) override;

  void buildHashTables() override;

  void reBuildHashFunction() override;

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

  void buildPatchMaps(std::tuple<uint32_t, uint32_t> next_kernel_size);

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