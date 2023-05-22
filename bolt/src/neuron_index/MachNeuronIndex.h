#pragma once

#include <bolt/src/neuron_index/NeuronIndex.h>
#include <dataset/src/mach/MachIndex.h>
#include <vector>

namespace thirdai::bolt::nn {

class MachNeuronIndex final : public NeuronIndex {
 public:
  void query(const BoltVector& input, BoltVector& output,
             const BoltVector* labels, uint32_t sparse_dim) const final;

  void buildIndex(const std::vector<float>& weights, uint32_t dim,
                  bool use_new_seed) final {
    (void)weights;
    (void)dim;
    (void)use_new_seed;
  }

  void autotuneForNewSparsity(uint32_t dim, uint32_t prev_dim, float sparsity,
                              bool experimental_autotune) final {
    (void)dim;
    (void)prev_dim;
    (void)sparsity;
    (void)experimental_autotune;
  }

  void summarize(std::ostream& summary) const final { summary << "mach_index"; }

 private:
  dataset::mach::MachIndexPtr _mach_index;

  std::vector<uint32_t> _rand_neurons;
};

}  // namespace thirdai::bolt::nn