#pragma once

#include <cereal/access.hpp>
#include <bolt/src/neuron_index/NeuronIndex.h>
#include <dataset/src/mach/MachIndex.h>
#include <memory>
#include <vector>

namespace thirdai::bolt::nn {

class MachNeuronIndex final : public NeuronIndex {
 public:
  explicit MachNeuronIndex(dataset::mach::MachIndexPtr mach_index)
      : _mach_index(std::move(mach_index)) {}

  static auto make(dataset::mach::MachIndexPtr mach_index) {
    return std::make_shared<MachNeuronIndex>(std::move(mach_index));
  }

  void query(const BoltVector& input, BoltVector& output,
             const BoltVector* labels) const final;

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

  void setNewIndex(dataset::mach::MachIndexPtr new_index) {
    _mach_index = std::move(new_index);
  }

 private:
  dataset::mach::MachIndexPtr _mach_index;

  std::vector<uint32_t> _rand_neurons;

  MachNeuronIndex() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::bolt::nn