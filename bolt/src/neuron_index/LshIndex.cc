#include "LshIndex.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/layers/SamplingConfig.h>
#include <hashing/src/HashUtils.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <utils/Random.h>
#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>

namespace thirdai::bolt {

LshIndex::LshIndex(uint32_t layer_dim, hashing::HashFunctionPtr hash_fn,
                   hashtable::SampledHashTablePtr hash_table)
    : _hash_fn(std::move(hash_fn)),
      _hash_table(std::move(hash_table)),
      _rand_neurons(layer_dim) {
  std::mt19937 rng(global_random::nextSeed());
  std::iota(_rand_neurons.begin(), _rand_neurons.end(), 0);
  std::shuffle(_rand_neurons.begin(), _rand_neurons.end(), rng);
}

void LshIndex::query(const BoltVector& input, BoltVector& output,
                     const BoltVector* labels) const {
  assert(!output.isDense());

  uint32_t sparse_dim = output.len;

  std::unordered_set<uint32_t> selected_neurons;

  uint32_t label_len = labels == nullptr || labels->isDense() ? 0 : labels->len;
  for (uint32_t i = 0; i < label_len; i++) {
    selected_neurons.insert(labels->active_neurons[i]);
  }

  std::vector<uint32_t> hashes(_hash_fn->numTables());
  if (input.isDense()) {
    _hash_fn->hashSingleDense(input.activations, input.len, hashes.data());
  } else {
    _hash_fn->hashSingleSparse(input.active_neurons, input.activations,
                               input.len, hashes.data());
  }

  if (_insert_labels_when_not_found) {
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

  if (selected_neurons.size() < sparse_dim) {
    // here we use hashes[0] as our random number because rand() is not thread
    // safe and we want to have deterministic sampling. We do the additional
    // hash on because the range of each hash from the lsh hash functions is
    // probably less than the layer dim.
    uint32_t rand_offset =
        hashing::simpleIntegerHash(hashes.at(0)) % _rand_neurons.size();
    while (selected_neurons.size() < sparse_dim) {
      selected_neurons.insert(_rand_neurons[rand_offset++]);
      rand_offset = rand_offset % _rand_neurons.size();
    }
  }

  uint32_t cnt = 0;
  for (uint32_t i = 0; i < label_len; i++) {
    if (cnt == sparse_dim) {
      break;
    }
    output.active_neurons[cnt++] = labels->active_neurons[i];
    selected_neurons.erase(labels->active_neurons[i]);
  }

  for (auto x : selected_neurons) {
    if (cnt == sparse_dim) {
      break;
    }
    assert(x < _rand_neurons.size());
    output.active_neurons[cnt++] = x;
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

void LshIndex::autotuneForNewSparsity(uint32_t dim, uint32_t prev_dim,
                                      float sparsity,
                                      bool experimental_autotune) {
  auto sampling_config =
      DWTASamplingConfig::autotune(dim, sparsity, experimental_autotune);

  _hash_fn = sampling_config->getHashFunction(prev_dim);

  _hash_table = sampling_config->getHashTable();
}

void LshIndex::summarize(std::ostream& summary) const {
  summary << "hash_function=" << _hash_fn->getName() << ", ";

  if (_hash_fn->getName() == "DWTA") {
    auto dwta_hasher =
        std::dynamic_pointer_cast<hashing::DWTAHashFunction>(_hash_fn);
    summary << "permutations= " << dwta_hasher->getNumPermutations() << ", "
            << "binsize= " << dwta_hasher->getBinsize() << ", "
            << "hashes_per_table= " << dwta_hasher->getHashesPerTable() << ", ";
  }
  _hash_table->summarize(summary);
}

ar::ConstArchivePtr LshIndex::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("hash_fn", _hash_fn->toArchive());
  map->set("hash_table", _hash_table->toArchive());
  map->set("rand_neurons", ar::vecU32(_rand_neurons));
  map->set("insert_labels_when_not_found",
           ar::boolean(_insert_labels_when_not_found));

  return map;
}

std::shared_ptr<LshIndex> LshIndex::fromArchive(const ar::Archive& archive) {
  return std::make_shared<LshIndex>(archive);
}

LshIndex::LshIndex(const ar::Archive& archive)
    : _hash_table(
          hashtable::SampledHashTable::fromArchive(*archive.get("hash_table"))),
      _rand_neurons(archive.getAs<ar::VecU32>("rand_neurons")),
      _insert_labels_when_not_found(
          archive.boolean("insert_labels_when_not_found")) {
  std::string hash_fn_type = archive.get("hash_fn")->str("type");

  if (hash_fn_type == hashing::DWTAHashFunction::type()) {
    _hash_fn = hashing::DWTAHashFunction::fromArchive(*archive.get("hash_fn"));
  } else {
    throw std::invalid_argument("Unsupported hash function type '" +
                                hash_fn_type + "'.");
  }
}

template void LshIndex::serialize(cereal::BinaryInputArchive&);
template void LshIndex::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void LshIndex::serialize(Archive& archive) {
  archive(cereal::base_class<NeuronIndex>(this), _hash_fn, _hash_table,
          _rand_neurons, _insert_labels_when_not_found);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::LshIndex,
                               "thirdai::bolt::nn::LshIndex")