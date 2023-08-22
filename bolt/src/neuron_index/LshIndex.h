#pragma once

#include <bolt/src/neuron_index/NeuronIndex.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace thirdai::bolt {

class LshIndex final : public NeuronIndex {
 public:
  LshIndex(uint32_t layer_dim, hashing::HashFunctionPtr hash_fn,
           hashtable::SampledHashTablePtr hash_table);

  static auto make(uint32_t layer_dim, hashing::HashFunctionPtr hash_fn,
                   hashtable::SampledHashTablePtr hash_table) {
    return std::make_shared<LshIndex>(layer_dim, std::move(hash_fn),
                                      std::move(hash_table));
  }

  std::vector<std::pair<uint32_t, std::vector<uint32_t>>> manualQuery(
      const BoltVector& input) const;

  void query(const BoltVector& input, BoltVector& output,
             const BoltVector* labels) const final;

  void buildIndex(const std::vector<float>& weights, uint32_t dim,
                  bool use_new_seed) final;

  void autotuneForNewSparsity(uint32_t dim, uint32_t prev_dim, float sparsity,
                              bool experimental_autotune) final;

  void summarize(std::ostream& summary) const final;

  const auto& hashFn() const { return _hash_fn; }

  const auto& hashTable() const { return _hash_table; }

  void insertLabelsIfNotFound() final { _insert_labels_when_not_found = true; }

  static auto cast(const NeuronIndexPtr& index) {
    return std::dynamic_pointer_cast<LshIndex>(index);
  }

 private:
  hashing::HashFunctionPtr _hash_fn;
  hashtable::SampledHashTablePtr _hash_table;

  std::vector<uint32_t> _rand_neurons;

  bool _insert_labels_when_not_found = false;

  LshIndex() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::bolt
