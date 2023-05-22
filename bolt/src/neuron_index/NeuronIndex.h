#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <unordered_set>

namespace thirdai::bolt::nn {

class NeuronIndex {
 public:
  virtual void query(const BoltVector& input, BoltVector& output,
                     const BoltVector* labels, uint32_t sparse_dim) const = 0;

  virtual void buildIndex(const std::vector<float>& weights, uint32_t dim,
                          bool use_new_seed) = 0;

  virtual void autotuneForNewSparsity(uint32_t dim, uint32_t prev_dim,
                                      float sparsity,
                                      bool experimental_autotune) = 0;

  virtual void summarize(std::ostream& summary) const = 0;
};

using NeuronIndexPtr = std::shared_ptr<NeuronIndex>;

}  // namespace thirdai::bolt::nn