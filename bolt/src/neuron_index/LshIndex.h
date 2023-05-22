#pragma once

#include <bolt/src/neuron_index/NeuronIndex.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <memory>
#include <random>

namespace thirdai::bolt::nn {

class LshIndex final : public NeuronIndex {
 public:
  LshIndex(uint32_t layer_dim, hashing::HashFunctionPtr hash_fn,
           hashtable::SampledHashTablePtr hash_table, std::random_device& rd);

  static auto make(uint32_t layer_dim, hashing::HashFunctionPtr hash_fn,
                   hashtable::SampledHashTablePtr hash_table,
                   std::random_device& rd) {
    return std::make_shared<LshIndex>(layer_dim, std::move(hash_fn),
                                      std::move(hash_table), rd);
  }

  void query(const BoltVector& input, BoltVector& output,
             const BoltVector* labels, uint32_t sparse_dim) const final;

  void buildIndex(const std::vector<float>& weights, uint32_t dim,
                  bool use_new_seed) final;

  void autotuneForNewSparsity(uint32_t dim, uint32_t prev_dim, float sparsity,
                              bool experimental_autotune) final;

  void summarize(std::ostream& summary) const final;

  const auto& hashFn() const { return _hash_fn; }

  const auto& hashTable() const { return _hash_table; }

  void insertLabelsIfNotFound() { _insert_labels_when_not_found = true; }

  static auto cast(const NeuronIndexPtr& index) {
    return std::dynamic_pointer_cast<LshIndex>(index);
  }

 private:
  hashing::HashFunctionPtr _hash_fn;
  hashtable::SampledHashTablePtr _hash_table;
  uint32_t _layer_dim;

  std::vector<uint32_t> _rand_neurons;

  bool _insert_labels_when_not_found = false;
};

}  // namespace thirdai::bolt::nn
