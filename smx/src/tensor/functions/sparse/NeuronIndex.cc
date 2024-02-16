#include "NeuronIndex.h"
#include <bolt/src/layers/SamplingConfig.h>
#include <memory>

namespace thirdai::smx {

LshIndex::LshIndex(const hashing::HashFunctionPtr& hash_fn,
                   size_t reservoir_size, const DenseTensorPtr& weight,
                   size_t updates_per_rebuild, size_t updates_per_new_hash_fn)
    : _weight(weight),
      _hash_fn(hash_fn),
      _hash_table(hash_fn->numTables(), reservoir_size, hash_fn->range()),
      _updates_per_rebuild(updates_per_rebuild),
      _updates_per_new_hash_fn(updates_per_new_hash_fn),
      _rand_neurons(weight->shape(0)),
      _frozen(false) {
  std::mt19937 rng(global_random::nextSeed());
  std::iota(_rand_neurons.begin(), _rand_neurons.end(), 0);
  std::shuffle(_rand_neurons.begin(), _rand_neurons.end(), rng);

  rebuild();
}

std::shared_ptr<LshIndex> LshIndex::autotune(size_t dim, size_t input_dim,
                                             float sparsity,
                                             const DenseTensorPtr& weight,
                                             size_t updates_per_rebuild,
                                             size_t updates_per_new_hash_fn) {
  // TODO(Nicholas): Move this autotuning code out of the SamplingConfig.
  auto config = bolt::DWTASamplingConfig::autotune(dim, sparsity, false);

  return make(config->getHashFunction(input_dim), config->reservoirSize(),
              weight, updates_per_rebuild, updates_per_new_hash_fn);
}

void LshIndex::query(const float* query, uint32_t* candidates,
                     size_t n_candidates, const uint32_t* force_select,
                     size_t n_force_select) {
  CHECK(force_select || n_force_select == 0,
        "n_force_select must be zero if force_select is null.");

  std::unordered_set<uint32_t> selected_neurons;

  if (force_select) {
    selected_neurons.insert(force_select, force_select + n_force_select);
  }

  std::vector<uint32_t> hashes(_hash_fn->numTables());
  _hash_fn->hashSingleDense(query, _weight->shape(1), hashes.data());

  _hash_table.queryBySet(hashes.data(), selected_neurons);

  if (_insert_missing_labels) {
    _hash_table.queryAndInsertForInference(hashes.data(), selected_neurons,
                                           n_candidates);
  } else {
    _hash_table.queryBySet(hashes.data(), selected_neurons);
  }

  if (selected_neurons.size() < n_candidates) {
    // here we use hashes[0] as our random number because rand() is not thread
    // safe and we want to have deterministic sampling. We do the additional
    // hash on because the range of each hash from the lsh hash functions is
    // probably less than the layer dim.
    size_t rand_offset =
        hashing::simpleIntegerHash(hashes.at(0)) % _rand_neurons.size();
    while (selected_neurons.size() < n_candidates) {
      selected_neurons.insert(_rand_neurons[rand_offset++]);
      rand_offset = rand_offset % _rand_neurons.size();
    }
  }

  size_t selected_idx = 0;
  for (size_t i = 0; i < n_force_select && selected_idx < n_candidates; i++) {
    candidates[selected_idx++] = force_select[i];
    selected_neurons.erase(force_select[i]);
  }
  for (auto x : selected_neurons) {
    if (selected_idx == n_candidates) {
      break;
    }
    candidates[selected_idx++] = x;
  }
}

void LshIndex::onUpdate() {
  _updates_since_rebuild++;
  _updates_per_new_hash_fn++;

  if (_frozen) {
    return;
  }

  if (_updates_since_new_hash_fn == _updates_per_new_hash_fn) {
    _updates_since_new_hash_fn = 0;
    _updates_since_rebuild = 0;

    _hash_fn = _hash_fn->copyWithNewSeeds();

    rebuild();
  } else if (_updates_since_rebuild == _updates_per_rebuild) {
    _updates_since_rebuild = 0;

    rebuild();
  }
}

void LshIndex::rebuild() {
  const float* weights = _weight->data<float>();

  size_t dim = _weight->shape(0);
  size_t input_dim = _weight->shape(1);
  size_t n_tables = _hash_table.numTables();

  std::vector<uint32_t> hashes(n_tables * dim);
#pragma omp parallel for default(none) \
    shared(n_tables, hashes, dim, input_dim, weights)
  for (uint64_t n = 0; n < dim; n++) {
    _hash_fn->hashSingleDense(weights + n * input_dim, input_dim,
                              hashes.data() + n * n_tables);
  }

  _hash_table.clearTables();
  _hash_table.insertSequential(dim, /*start=*/0, hashes.data());
}

}  // namespace thirdai::smx