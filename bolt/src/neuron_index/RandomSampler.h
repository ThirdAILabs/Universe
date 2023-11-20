#pragma once

#include <cereal/access.hpp>
#include <bolt/src/neuron_index/NeuronIndex.h>
#include <archive/src/Archive.h>
#include <random>

namespace thirdai::bolt {

class RandomSampler final : public NeuronIndex {
 public:
  explicit RandomSampler(uint32_t layer_dim);

  explicit RandomSampler(const ar::Archive& archive);

  static auto make(uint32_t layer_dim) {
    return std::make_shared<RandomSampler>(layer_dim);
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

  void summarize(std::ostream& summary) const final { summary << "random"; }

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<RandomSampler> fromArchive(const ar::Archive& archive);

  static auto cast(const NeuronIndexPtr& index) {
    return std::dynamic_pointer_cast<RandomSampler>(index);
  }

  static std::string type() { return "random_sampler"; }

 private:
  std::vector<uint32_t> _rand_neurons;

  RandomSampler() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::bolt