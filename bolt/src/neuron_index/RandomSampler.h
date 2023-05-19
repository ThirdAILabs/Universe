#pragma once

#include <bolt/src/neuron_index/NeuronIndex.h>

namespace thirdai::bolt::nn {

class RandomSampler final : public NeuronIndex {
 public:
  void query(const BoltVector& input,
             std::unordered_set<uint32_t>& selected_neurons,
             uint32_t sparse_dim) const final {
    (void)input;
    (void)sparse_dim;
    (void)selected_neurons;
  }

  void buildIndex(const std::vector<float>& weights, uint32_t dim,
                  bool use_new_seed) final {
    (void)weights;
    (void)dim;
    (void)use_new_seed;
  }

  void updateSparsity(uint32_t dim, uint32_t prev_dim, float sparsity,
                      bool experimental_autotune) final {
    (void)dim;
    (void)prev_dim;
    (void)sparsity;
    (void)experimental_autotune;
  }
};

}  // namespace thirdai::bolt::nn