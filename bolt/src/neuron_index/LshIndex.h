#pragma once

#include <bolt/src/neuron_index/NeuronIndex.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <memory>

namespace thirdai::bolt::nn {

class LshIndex final : public NeuronIndex {
 public:
  LshIndex(hashing::HashFunctionPtr hash_fn,
           hashtable::SampledHashTablePtr hash_table)
      : _hash_fn(std::move(hash_fn)), _hash_table(std::move(hash_table)) {}

  static auto make(hashing::HashFunctionPtr hash_fn,
                   hashtable::SampledHashTablePtr hash_table) {
    return std::make_unique<LshIndex>(std::move(hash_fn),
                                      std::move(hash_table));
  }

  void query(const BoltVector& input,
             std::unordered_set<uint32_t>& selected_neurons,
             uint32_t sparse_dim) const final;

  void buildIndex(const std::vector<float>& weights, uint32_t dim,
                  bool use_new_seed) final;

  void updateSparsity(uint32_t dim, uint32_t prev_dim, float sparsity,
                      bool experimental_autotune) final;

  static LshIndex* cast(const NeuronIndexPtr& index) {
    return dynamic_cast<LshIndex*>(index.get());
  }

 private:
  hashing::HashFunctionPtr _hash_fn;
  hashtable::SampledHashTablePtr _hash_table;

  bool _freeze_with_insertions = false;
};

}  // namespace thirdai::bolt::nn
