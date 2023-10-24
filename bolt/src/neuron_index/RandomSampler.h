#pragma once

#include <cereal/access.hpp>
#include <bolt/src/neuron_index/NeuronIndex.h>
#include <proto/neuron_index.pb.h>
#include <random>

namespace thirdai::bolt {

class RandomSampler final : public NeuronIndex {
 public:
  explicit RandomSampler(uint32_t layer_dim);

  explicit RandomSampler(const proto::bolt::RandomNeuronIndex& rand_proto);

  static auto make(uint32_t layer_dim) {
    return std::make_shared<RandomSampler>(layer_dim);
  }

  static auto fromProto(const proto::bolt::RandomNeuronIndex& rand_proto) {
    return std::make_shared<RandomSampler>(rand_proto);
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

  proto::bolt::NeuronIndex* toProto() const;

  static auto cast(const NeuronIndexPtr& index) {
    return std::dynamic_pointer_cast<RandomSampler>(index);
  }

 private:
  std::vector<uint32_t> _rand_neurons;

  RandomSampler() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::bolt