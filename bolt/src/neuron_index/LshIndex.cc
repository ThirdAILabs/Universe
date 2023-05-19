#include "LshIndex.h"
#include <bolt/src/layers/SamplingConfig.h>

namespace thirdai::bolt::nn {

void LshIndex::query(const BoltVector& input,
                     std::unordered_set<uint32_t>& selected_neurons,
                     uint32_t sparse_dim) const {
  std::vector<uint32_t> hashes(_hash_fn->numTables());
  if (input.isDense()) {
    _hash_fn->hashSingleDense(input.activations, input.len, hashes.data());
  } else {
    _hash_fn->hashSingleSparse(input.active_neurons, input.activations,
                               input.len, hashes.data());
  }

  if (_freeze_with_insertions) {
    /**
     * QueryBySet just returns a set of the elements in the given buckets of
     * the hash table.
     *
     * QueryAndInsertForInference returns the set of elements in the given
     * buckets but will also insert the labels (during training only) for the
     * vector into the buckets the vector maps to if they are not already
     * present in the buckets. The intuition is that during sparse inference
     * this will help force the hash tables to map vectors towards buckets
     * that contain their correct labels. This is specific to the output
     * layer.
     */
    _hash_table->queryAndInsertForInference(hashes.data(), selected_neurons,
                                            sparse_dim);
  } else {
    _hash_table->queryBySet(hashes.data(), selected_neurons);
  }
}

void LshIndex::buildIndex(const std::vector<float>& weights, uint32_t dim,
                          bool use_new_seed) {
  if (use_new_seed) {
    _hash_fn = _hash_fn->copyWithNewSeeds();
  }
  uint32_t prev_dim = weights.size() / dim;

  uint32_t num_tables = _hash_table->numTables();
  std::vector<uint32_t> hashes(num_tables * dim);
#pragma omp parallel for default(none) \
    shared(num_tables, hashes, dim, prev_dim, weights)
  for (uint64_t n = 0; n < dim; n++) {
    _hash_fn->hashSingleDense(weights.data() + n * prev_dim, prev_dim,
                              hashes.data() + n * num_tables);
  }

  _hash_table->clearTables();
  _hash_table->insertSequential(dim, 0, hashes.data());
}

void LshIndex::updateSparsity(uint32_t dim, uint32_t prev_dim, float sparsity,
                              bool experimental_autotune) {
  auto sampling_config =
      DWTASamplingConfig::autotune(dim, sparsity, experimental_autotune);

  _hash_fn = sampling_config->getHashFunction(prev_dim);

  _hash_table = sampling_config->getHashTable();
}

}  // namespace thirdai::bolt::nn