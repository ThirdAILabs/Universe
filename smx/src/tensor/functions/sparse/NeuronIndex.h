#pragma once

#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <smx/src/tensor/DenseTensor.h>
#include <algorithm>

namespace thirdai::smx {

class NeuronIndex {
 public:
  virtual void query(const float* query, uint32_t* candidates,
                     size_t n_candidates, const uint32_t* force_select,
                     size_t n_force_select) = 0;

  virtual void onUpdate() = 0;

  virtual void freeze() {}
};

using NeuronIndexPtr = std::shared_ptr<NeuronIndex>;

class LshIndex final : public NeuronIndex {
 public:
  LshIndex(const hashing::HashFunctionPtr& hash_fn, size_t reservoir_size,
           const DenseTensorPtr& weight, size_t updates_per_rebuild,
           size_t updates_per_new_hash_fn);

  static auto make(const hashing::HashFunctionPtr& hash_fn,
                   size_t reservoir_size, const DenseTensorPtr& weight,
                   size_t _updates_per_rebuild,
                   size_t _updates_per_new_hash_fn) {
    return std::make_shared<LshIndex>(hash_fn, reservoir_size, weight,
                                      _updates_per_rebuild,
                                      _updates_per_new_hash_fn);
  }

  static std::shared_ptr<LshIndex> autotune(size_t dim, size_t input_dim,
                                            float sparsity,
                                            const DenseTensorPtr& weight,
                                            size_t updates_per_rebuild,
                                            size_t updates_per_new_hash_fn);

  void query(const float* query, uint32_t* candidates, size_t n_candidates,
             const uint32_t* force_select, size_t n_force_select) final;

  void onUpdate() final;

  void freeze() final { _frozen = true; }

  void rebuild();

 private:
  DenseTensorPtr _weight;

  hashing::HashFunctionPtr _hash_fn;
  hashtable::SampledHashTable _hash_table;

  size_t _updates_since_rebuild = 0;
  size_t _updates_since_new_hash_fn = 0;

  size_t _updates_per_rebuild;
  size_t _updates_per_new_hash_fn;

  std::vector<uint32_t> _rand_neurons;

  bool _frozen = false;
};

}  // namespace thirdai::smx